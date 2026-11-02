"""Shared WALL learner for VDN, QMIX and QPLEX.

Defender actions are the issued actions in the LPA-induced MDP. Dynamics sees
executed (possibly attacked) actions. Auxiliary labels are built lazily from the
current policy, rather than storing stale hypothetical groups/recurrent states.
"""
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import RMSprop

from components.wolfpack import joint_q
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.dmaq_general import DMAQer
from .wall_auxiliary import WALLAuxiliaryLearner, transition_mask


def _load(path, device="cpu", weights_only=False):
    # Only explicitly selected local training checkpoints are loaded here.
    return torch.load(str(path), map_location=device, weights_only=weights_only)


def _grad_scaler(enabled):
    # PyTorch 2.5 public API, with compatibility for the measured 2.0 stack.
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


class QLearner:
    checkpoint_version = 2

    def __init__(self, mac, scheme, logger, args, qdifference_transformer, planning_transformer):
        self.args, self.mac, self.logger = args, mac, logger
        factories = {"vdn": VDNMixer, "qmix": lambda: QMixer(args), "dmaq": lambda: DMAQer(args)}
        if args.mixer not in factories:
            raise ValueError("Optimized WALL supports mixer='vdn', 'qmix', or 'dmaq'.")
        self.mixer = factories[args.mixer]()
        self.params = list(mac.parameters()) + list(self.mixer.parameters())
        self.optimiser = RMSprop(self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.target_mac = copy.deepcopy(mac)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.qdiff_transformer = qdifference_transformer
        self.planning_transformer = planning_transformer
        self.optimizer_qdiff = torch.optim.Adam(
            self.qdiff_transformer.parameters(),
            lr=args.lr if getattr(args, "qdiff_lr", None) is None else args.qdiff_lr)
        self.optimizer_planning = torch.optim.Adam(
            self.planning_transformer.parameters(),
            lr=args.lr if getattr(args, "planner_lr", None) is None else args.planner_lr)
        self.last_target_update_episode = 0
        self.wall_updates = 0
        self.log_stats_t = -getattr(args, "learner_log_interval", 10000) - 1
        self.scaler = _grad_scaler(enabled=False)
        self.auxiliary = WALLAuxiliaryLearner(self)
        self.last_metrics = {}
        self._freeze_targets()

    def optimizer_metrics(self):
        """Report effective optimizer values, including values restored on resume."""
        return {"lr_defender": float(self.optimiser.param_groups[0]["lr"]),
                "lr_planner": float(self.optimizer_planning.param_groups[0]["lr"]),
                "lr_qdiff": float(self.optimizer_qdiff.param_groups[0]["lr"])}

    def _log_optimizer_settings(self):
        console = getattr(self.logger, "console_logger", None)
        if console is not None:
            console.info("Effective optimizer learning rates: %s", self.optimizer_metrics())

    def _freeze_targets(self):
        self.target_mac.agent.eval()
        self.target_mac.agent.requires_grad_(False)
        self.target_mixer.eval().requires_grad_(False)

    def to(self, device):
        device = torch.device(device)
        self.mac.to(device)
        self.target_mac.to(device)
        self.mixer.to(device)
        self.target_mixer.to(device)
        self.qdiff_transformer.to(device)
        self.planning_transformer.to(device)
        for opt in (self.optimiser, self.optimizer_qdiff, self.optimizer_planning):
            _optimizer_to(opt, device)
        amp_fp16 = (device.type == "cuda" and getattr(self.args, "use_amp", False)
                    and getattr(self.args, "amp_dtype", "bfloat16") == "float16")
        self.scaler = _grad_scaler(enabled=amp_fp16)
        self.auxiliary.configure()
        return self

    def cuda(self):
        return self.to("cuda")

    def _mix(self, q, states, actions, available, target=False):
        shape = q.shape[:-2]
        mixer = self.target_mixer if target else self.mixer
        result = joint_q(mixer, self.args, q.reshape(-1, self.args.n_agents, self.args.n_actions),
                         states.reshape(-1, states.shape[-1]), actions.reshape(-1, self.args.n_agents),
                         available.reshape(-1, self.args.n_agents, self.args.n_actions),
                         detach_advantage=True)
        return result.reshape(*shape, 1)

    def train(self, batch, t_env, episode_num):
        started = time.perf_counter()
        valid = transition_mask(batch)
        if valid.numel() == 0 or not valid.any():
            self.last_metrics = {"skipped_empty_batch": 1.0}
            return self.last_metrics
        self.mac.agent.train()
        self.mixer.train()
        self.optimiser.zero_grad(set_to_none=True)
        with torch.profiler.record_function("wall/td_update"):
            # All training timesteps share one fused GRU call; recurrent outputs
            # are reused by the imagined rollout instead of replay hidden states.
            online, hidden_after = self.mac.forward_sequence(batch, return_hidden=True)
            chosen = self._mix(online[:, :-1], batch["state"][:, :-1],
                               batch["actions"][:, :-1].squeeze(-1), batch["avail_actions"][:, :-1])
            with torch.no_grad():
                target = self.target_mac.forward_sequence(batch)[:, 1:]
                available = batch["avail_actions"][:, 1:].bool()
                # All-zero masks occur in padding; safe finite target values are
                # masked out by `valid` later instead of propagating inf*0.
                safe_available = available | (~available.any(-1, keepdim=True) & F.one_hot(
                    torch.zeros_like(available[..., 0], dtype=torch.long), self.args.n_actions).bool())
                chooser = online[:, 1:].detach() if self.args.double_q else target
                next_actions = chooser.masked_fill(~safe_available, -torch.inf).argmax(-1)
                bootstrap = self._mix(target, batch["state"][:, 1:], next_actions, safe_available, target=True)
                targets = batch["reward"][:, :-1] + self.args.gamma * (
                    1 - batch["terminated"][:, :-1].float()) * bootstrap
            td_error = chosen - targets
            loss = (td_error.squeeze(-1).square() * valid).sum() / valid.sum()
            # Release TD activations before world-model updates, while delaying
            # the optimizer step so all auxiliary targets use the same policy.
            q_detached, h_detached = online.detach(), hidden_after.detach()
            loss.backward()
            del online, hidden_after
        aux_batch = batch
        aux_q, aux_h, aux_valid = q_detached, h_detached, valid
        aux_size = min(batch.batch_size, int(getattr(self.args, "planner_batch_size", batch.batch_size)))
        if aux_size < batch.batch_size:
            indices = torch.randperm(batch.batch_size, device=valid.device)[:aux_size]
            # EpisodeBatch indexing supports numpy/CPU index tensors; tensor
            # observations may reside on CUDA, which supports CPU long indices.
            aux_batch = batch[indices.cpu()]
            aux_q, aux_h, aux_valid = q_detached[indices], h_detached[indices], valid[indices]
        aux_metrics = (self.auxiliary.train(aux_batch, aux_q, aux_h, aux_valid)
                       if getattr(self.args, "train_attack_predictors", True) else {})
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        self.wall_updates += 1
        mode = getattr(self.args, "target_update_mode", "hard")
        if mode == "ema":
            self._update_targets(tau=float(getattr(self.args, "target_update_tau", 0.005)))
        elif mode == "hard":
            if episode_num - self.last_target_update_episode >= self.args.target_update_interval:
                self._update_targets()
                self.last_target_update_episode = episode_num
        else:
            raise ValueError("target_update_mode must be 'hard' or 'ema'")
        metrics = {"loss_td": loss.detach(), "grad_norm": grad_norm.detach(),
                   "td_error_abs": (td_error.detach().squeeze(-1).abs() * valid).sum() / valid.sum(),
                   "q_taken_mean": (chosen.detach().squeeze(-1) * valid).sum() / valid.sum(),
                   "target_mean": (targets.squeeze(-1) * valid).sum() / valid.sum(), **aux_metrics}
        # Host values only once per logged update. GPU duration needs profiler
        # events; this timer is deliberately labelled host wall time.
        if t_env - self.log_stats_t >= getattr(self.args, "learner_log_interval", 10000):
            keys = list(metrics)
            values = torch.stack([metrics[k].float() for k in keys]).detach().cpu().tolist()
            self.last_metrics = dict(zip(keys, values))
            self.last_metrics.update(self.optimizer_metrics())
            self.last_metrics["learner_host_wall_seconds"] = time.perf_counter() - started
            if self.logger is not None:
                for key, value in self.last_metrics.items():
                    self.logger.log_stat(key, value, t_env)
            self.log_stats_t = t_env
        return self.last_metrics

    @torch.no_grad()
    def _update_targets(self, tau=1.0):
        if not 0 < tau <= 1:
            raise ValueError("target_update_tau must lie in (0, 1].")
        for online, target in ((self.mac.agent, self.target_mac.agent), (self.mixer, self.target_mixer)):
            for op, tp in zip(online.parameters(), target.parameters()):
                tp.lerp_(op, tau)
            for ob, tb in zip(online.buffers(), target.buffers()):
                tb.copy_(ob)
        self._freeze_targets()

    def _fingerprint(self):
        names = ("mixer", "n_agents", "n_actions", "state_shape", "obs_shape", "rnn_hidden_dim",
                 "trans_embed_dim", "num_trans_layer", "num_trans_head", "trans_input_len", "attack_period")
        fingerprint = {k: getattr(self.args, k) for k in names}
        fingerprint["map_name"] = getattr(self.args, "env_args", {}).get("map_name")
        fingerprint["trans_ff_multiplier"] = getattr(self.args, "trans_ff_multiplier", 4)
        return fingerprint

    def save_models(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.mac.save_models(str(path))
        torch.save(self.mixer.state_dict(), path / "mixer.th")
        torch.save(self.optimiser.state_dict(), path / "opt.th")
        checkpoint = {
            "format_version": self.checkpoint_version, "fingerprint": self._fingerprint(),
            "agent": self.mac.agent.state_dict(), "mixer": self.mixer.state_dict(),
            "target_agent": self.target_mac.agent.state_dict(), "target_mixer": self.target_mixer.state_dict(),
            "planning": self.planning_transformer.state_dict(), "qdiff": self.qdiff_transformer.state_dict(),
            "optimizer": self.optimiser.state_dict(), "optimizer_planning": self.optimizer_planning.state_dict(),
            "optimizer_qdiff": self.optimizer_qdiff.state_dict(), "scaler": self.scaler.state_dict(),
            "wall_updates": self.wall_updates, "last_target_update_episode": self.last_target_update_episode,
            "log_stats_t": self.log_stats_t,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }
        temp = path / "learner.th.tmp"
        torch.save(checkpoint, temp)
        temp.replace(path / "learner.th")

    def load_models(self, path, resume=False):
        path = Path(path)
        device = next(self.mac.parameters()).device
        if not resume:
            # Defender warm-start intentionally does not load incompatible legacy
            # planner architecture. Existing GRUCell keys are migrated by RNNAgent.
            self.mac.load_models(str(path))
            self.mixer.load_state_dict(_load(path / "mixer.th", device, weights_only=True))
            if getattr(self.args, "load_pretrain_optimizer", True) and (path / "opt.th").is_file():
                # A warm-start imports RMSProp moments, but this experiment's
                # hyperparameters must win over the pretrained param_groups.
                # Full resume below deliberately restores checkpoint settings.
                configured_groups = [{k: copy.deepcopy(v) for k, v in group.items() if k != "params"}
                                     for group in self.optimiser.param_groups]
                self.optimiser.load_state_dict(_load(path / "opt.th", device, weights_only=True))
                for group, configured in zip(self.optimiser.param_groups, configured_groups):
                    group.update(configured)
                _optimizer_to(self.optimiser, device)
            self._update_targets()
            self._log_optimizer_settings()
            return
        checkpoint_path = path / "learner.th"
        if not checkpoint_path.is_file():
            raise ValueError("Full resume requires an optimized learner.th checkpoint. "
                             "Use pretrain_dir for a legacy defender warm-start; legacy predictors are incompatible.")
        checkpoint = _load(checkpoint_path, "cpu")
        if checkpoint.get("format_version") != self.checkpoint_version:
            raise ValueError("Unsupported optimized checkpoint format version.")
        if checkpoint.get("fingerprint") != self._fingerprint():
            raise ValueError("Checkpoint architecture/map differs from this run; use matching configuration.")
        self.mac.agent.load_state_dict(checkpoint["agent"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_mac.agent.load_state_dict(checkpoint["target_agent"])
        self.target_mixer.load_state_dict(checkpoint["target_mixer"])
        self.planning_transformer.load_state_dict(checkpoint["planning"])
        self.qdiff_transformer.load_state_dict(checkpoint["qdiff"])
        self.optimiser.load_state_dict(checkpoint["optimizer"])
        self.optimizer_planning.load_state_dict(checkpoint["optimizer_planning"])
        self.optimizer_qdiff.load_state_dict(checkpoint["optimizer_qdiff"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        for opt in (self.optimiser, self.optimizer_planning, self.optimizer_qdiff):
            _optimizer_to(opt, device)
        self.wall_updates = checkpoint["wall_updates"]
        self.last_target_update_episode = checkpoint["last_target_update_episode"]
        self.log_stats_t = checkpoint.get("log_stats_t", -1)
        self._freeze_targets()
        self._log_optimizer_settings()
        rng = checkpoint["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            if len(rng["cuda"]) != torch.cuda.device_count():
                raise ValueError("CUDA RNG checkpoint device count does not match visible devices.")
            torch.cuda.set_rng_state_all(rng["cuda"])

    def load_attackers(self, path):
        """Compatibility entry point: new predictor architectures only."""
        checkpoint = _load(Path(path) / "learner.th", "cpu")
        if checkpoint.get("format_version") != self.checkpoint_version:
            raise ValueError("Legacy attacker weights cannot be loaded into the causal optimized models.")
        self.planning_transformer.load_state_dict(checkpoint["planning"])
        self.qdiff_transformer.load_state_dict(checkpoint["qdiff"])
        self.optimizer_planning.load_state_dict(checkpoint["optimizer_planning"])
        self.optimizer_qdiff.load_state_dict(checkpoint["optimizer_qdiff"])
        device = next(self.mac.parameters()).device
        _optimizer_to(self.optimizer_planning, device)
        _optimizer_to(self.optimizer_qdiff, device)
