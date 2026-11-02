"""Batched world-model training and paper-defined Wolfpack effect distillation.

The Q-difference network predicts E_i[Delta Q_WP | state history], where the
initial agent i is uniform. One independent sampled initial agent per target is
an unbiased Monte Carlo label; no agent-conditioned horizon labels are mixed.
"""
from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F

from components.wolfpack import attack_actions, joint_q, select_followup
from utils.sequence import gather_history


def transition_mask(batch):
    """A transition needs both observations, and must not follow a true terminal."""
    filled = batch["filled"].squeeze(-1).bool()
    terminated = batch["terminated"][:, :-1].squeeze(-1).bool()
    alive = torch.cat((torch.ones_like(terminated[:, :1]),
                       (~terminated[:, :-1]).cumprod(1).bool()), dim=1)
    return filled[:, :-1] & filled[:, 1:] & alive


def horizon_targets(effects, valid, horizon):
    """Align origin+horizon within each sampled span, excluding padded pairs.

    effects/valid: [batch, spans, time]. Returns [batch, spans, time, horizon].
    """
    span = effects.shape[-1]
    offsets = torch.arange(span, device=effects.device)[:, None] + torch.arange(
        horizon, device=effects.device)[None, :]
    indices = offsets.clamp_max(span - 1)
    target = effects[..., indices]
    mask = valid[..., None] & valid[..., indices] & (offsets < span)
    return target, mask


@contextmanager
def evaluating(*modules):
    flags = [m.training for m in modules]
    try:
        for m in modules:
            m.eval()
        yield
    finally:
        for m, flag in zip(modules, flags):
            m.train(flag)


def _legal_mask(logits):
    """Prediction may contain no legal action: retain the most likely action."""
    available = logits >= 0
    fallback = F.one_hot(logits.argmax(-1), logits.shape[-1]).bool()
    return available | (~available.any(-1, keepdim=True) & fallback)


class WALLAuxiliaryLearner:
    def __init__(self, owner):
        self.owner = owner
        self.args = owner.args
        self.planner = owner.planning_transformer
        self.qdiff = owner.qdiff_transformer
        self._planner_call = self.planner
        self._qdiff_call = self.qdiff
        self.compiled = False

    def configure(self):
        # Keep unwrapped modules for stable checkpoint names and runner sharing.
        if getattr(self.args, "compile_models", False) and not self.compiled:
            if not hasattr(torch, "compile"):
                raise RuntimeError("compile_models requires a torch.compile-capable PyTorch.")
            self._planner_call = torch.compile(self.planner, dynamic=True)
            self._qdiff_call = torch.compile(self.qdiff, dynamic=True)
            self.compiled = True

    def autocast(self):
        device = next(self.planner.parameters()).device
        if not getattr(self.args, "use_amp", False) or device.type != "cuda":
            return nullcontext()
        dtype = torch.bfloat16 if getattr(self.args, "amp_dtype", "bfloat16") == "bfloat16" else torch.float16
        return torch.autocast("cuda", dtype=dtype)

    def _step(self, loss, optimizer, parameters):
        optimizer.zero_grad(set_to_none=True)
        self.owner.scaler.scale(loss).backward()
        self.owner.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, self.args.grad_norm_clip)
        self.owner.scaler.step(optimizer)

    def train(self, batch, q_sequence, hidden_after, valid):
        args = self.args
        device = batch["state"].device
        b, t = valid.shape
        if t == 0 or not valid.any():
            return {}
        spans = int(getattr(args, "planner_origins_per_episode", 1))
        context = int(args.trans_input_len)
        span = int(getattr(args, "planner_train_span", None) or max(context, args.attack_period))
        if span < args.attack_period:
            raise ValueError("planner_train_span must cover every attack_period output")
        # Choose within each episode's own valid length, not the longest episode.
        lengths = valid.long().sum(1)
        possible_starts = (lengths - span + 1).clamp_min(1)
        starts = (torch.rand(b, spans, device=device) * possible_starts[:, None]).long()
        endpoints = (starts[..., None] + torch.arange(span, device=device)).reshape(b, -1)
        span_ends = starts + (lengths[:, None] - starts).clamp(min=1, max=span) - 1
        state = batch["state"][:, :-1]
        obs = batch["obs"][:, :-1]
        executed = batch["forced_actions_onehot"][:, :-1]
        world_histories, history_valid, timestamps = gather_history(state, span_ends, context, valid)
        obs_histories = gather_history(obs, span_ends, context, valid)[0]
        action_histories = gather_history(executed, span_ends, context, valid)[0]
        state_targets = gather_history(batch["state"][:, 1:], span_ends, context, valid)[0]
        obs_targets = gather_history(batch["obs"][:, 1:], span_ends, context, valid)[0]
        avail_targets = gather_history(batch["avail_actions"][:, 1:].float(), span_ends, context, valid)[0]
        terminal_targets = gather_history(batch["terminated"][:, :-1].float(), span_ends, context, valid)[0]
        mask = history_valid.flatten(0, 1)
        self.planner.train()
        with torch.profiler.record_function("wall/planner_update"), self.autocast():
            ps, po, pa, pt = self._planner_call(
                world_histories.flatten(0, 1), obs_histories.flatten(0, 1),
                action_histories.flatten(0, 1), timesteps=timestamps.flatten(0, 1),
                valid_mask=mask, return_aux=True)
            # Reductions remain FP32 under autocast.
            state_error = (ps.float() - state_targets.flatten(0, 1)).square().mean(-1)
            obs_error = (po.float() - obs_targets.flatten(0, 1)).square().mean((-1, -2))
            denominator = mask.sum().clamp_min(1)
            state_loss = (state_error * mask).sum() / denominator
            obs_loss = (obs_error * mask).sum() / denominator
            avail_error = F.binary_cross_entropy_with_logits(
                pa.float(), avail_targets.flatten(0, 1), reduction="none").mean((-1, -2))
            terminal_error = F.binary_cross_entropy_with_logits(
                pt.float(), terminal_targets.flatten(0, 1), reduction="none").squeeze(-1)
            avail_loss = (avail_error * mask).sum() / denominator
            terminal_loss = (terminal_error * mask).sum() / denominator
            aux = (avail_loss if getattr(args, "predict_availability", True) else 0) + (
                terminal_loss if getattr(args, "predict_termination", True) else 0)
            planner_loss = float(getattr(args, "planner_loss_weight", 1.0)) * (
                float(getattr(args, "planner_state_loss_weight", 1.0)) * state_loss +
                float(getattr(args, "planner_obs_loss_weight", 1.0)) * obs_loss +
                float(getattr(args, "world_model_aux_weight", 0.1)) * aux)
        self._step(planner_loss, self.owner.optimizer_planning, self.planner.parameters())

        with torch.profiler.record_function("wall/counterfactual_targets"):
            effects, effect_valid = self.attack_effects(batch, q_sequence, hidden_after, valid, endpoints)
        effects = effects.reshape(b, spans, span)
        effect_valid = effect_valid.reshape(b, spans, span)
        targets, pair_valid = horizon_targets(effects, effect_valid, args.attack_period)
        histories, valid_histories, times = gather_history(state, endpoints, context, valid)
        self.qdiff.train()
        with torch.profiler.record_function("wall/qdiff_update"), self.autocast():
            predictions = self._qdiff_call(times.flatten(0, 1), histories.flatten(0, 1),
                                           valid_mask=valid_histories.flatten(0, 1), last_only=True)[:, -1]
            predictions = predictions.reshape(b, spans, span, args.attack_period).float()
            qdiff_loss = float(getattr(args, "qdifference_loss_weight", 1.0)) * (
                ((predictions - targets).square() * pair_valid).sum() / pair_valid.sum().clamp_min(1))
        self._step(qdiff_loss, self.owner.optimizer_qdiff, self.qdiff.parameters())
        self.owner.scaler.update()
        return {"loss_planning": planner_loss.detach(), "loss_state": state_loss.detach(),
                "loss_obs": obs_loss.detach(), "loss_availability": avail_loss.detach(),
                "loss_termination": terminal_loss.detach(), "loss_qdiff": qdiff_loss.detach(),
                "qdiff_target_mean": (effects.sum() / effect_valid.sum().clamp_min(1)).detach(),
                "qdiff_valid_pairs": pair_valid.sum().detach()}

    @torch.no_grad()
    def attack_effects(self, batch, q_sequence, hidden_after, valid, endpoints):
        """Sum SAME-STATE normal/attacked Q gaps along an attacked trajectory.

        Only current-policy recurrent states are used. Sampled replay hidden
        states and recorded counterfactual labels are intentionally unnecessary.
        """
        args, mac, mixer = self.args, self.owner.mac, self.owner.mixer
        context = int(args.trans_input_len)
        b, k = endpoints.shape
        t = valid.shape[1]
        rows = torch.arange(b, device=endpoints.device)[:, None].expand(b, k)
        safe = endpoints.clamp(0, t - 1)
        endpoint_valid = (endpoints >= 0) & (endpoints < t) & valid[rows, safe]
        state_hist, history_valid, times = gather_history(batch["state"][:, :-1], endpoints, context, valid)
        obs_hist = gather_history(batch["obs"][:, :-1], endpoints, context, valid)[0]
        action_hist = gather_history(batch["forced_actions_onehot"][:, :-1], endpoints, context, valid)[0]
        state_hist = state_hist.flatten(0, 1)
        obs_hist = obs_hist.flatten(0, 1)
        action_hist = action_hist.flatten(0, 1)
        history_valid = history_valid.flatten(0, 1)
        times = times.flatten(0, 1)
        q = q_sequence[rows, safe].reshape(-1, args.n_agents, args.n_actions).float()
        hidden = hidden_after[rows, safe].reshape(-1, args.n_agents, args.rnn_hidden_dim)
        avail = batch["avail_actions"][rows, safe].reshape(-1, args.n_agents, args.n_actions).bool()
        # Padded endpoints need safe softmax/argmax even though their losses are masked.
        avail = avail | (~avail.any(-1, keepdim=True) & F.one_hot(
            torch.zeros_like(avail[..., 0], dtype=torch.long), args.n_actions).bool())
        normal_actions = q.masked_fill(~avail, -torch.inf).argmax(-1)
        initial = torch.randint(args.n_agents, (b * k,), device=q.device)
        target_mask = F.one_hot(initial, args.n_agents).bool()
        attacked = attack_actions(mixer, args, q, state_hist[:, -1], normal_actions, avail, target_mask)
        active = endpoint_valid.flatten() & (normal_actions != attacked).any(-1)
        # Empty/no-op starts contribute exactly zero. Compact after sampling the
        # initial agents so their uniform distribution is unaffected.
        active_rows = active.nonzero(as_tuple=False).squeeze(-1)
        output = torch.zeros(b * k, device=q.device)
        if active_rows.numel() == 0:
            return output.reshape(b, k), endpoint_valid
        state_hist, obs_hist, action_hist = (x[active_rows] for x in (state_hist, obs_hist, action_hist))
        history_valid, times = history_valid[active_rows], times[active_rows]
        q, hidden, avail = (x[active_rows] for x in (q, hidden, avail))
        normal_actions, attacked = normal_actions[active_rows], attacked[active_rows]
        initial, target_mask = initial[active_rows], target_mask[active_rows]
        active_endpoints = endpoints.flatten()[active_rows]
        # select_followup temporarily enables input-Q gradients, without parameter .grad.
        group = select_followup(mixer, args, q, state_hist[:, -1], normal_actions,
                                attacked, avail, initial, obs_hist[:, -1])
        followup_counts = torch.zeros_like(target_mask, dtype=torch.long)
        if group.shape[-1]:
            followup_counts.scatter_add_(1, group.clamp_min(0), (group >= 0).long())
        followup_mask = followup_counts > 0
        cumulative = torch.zeros(active_rows.numel(), device=q.device)
        weights = torch.ones_like(cumulative)
        # Horizon is capped by the known maximum episode limit, never by a replay
        # trajectory's future termination (that trajectory followed different actions).
        max_episode_steps = int(getattr(args, "episode_limit", getattr(args, "env_max_timestep", 300)))
        with evaluating(self.planner):
            for step in range(int(args.attack_duration) + 1):
                state_now = state_hist[:, -1]
                gap = joint_q(mixer, args, q, state_now, normal_actions, avail) - joint_q(
                    mixer, args, q, state_now, attacked, avail)
                within_limit = (active_endpoints + step < max_episode_steps)
                cumulative += weights * within_limit * gap
                if step == int(args.attack_duration):
                    break
                action_hist[:, -1] = F.one_hot(attacked, args.n_actions).to(action_hist.dtype)
                with self.autocast():
                    ps, po, pa, pt = self._planner_call(state_hist, obs_hist, action_hist,
                                                       timesteps=times, valid_mask=history_valid, return_aux=True, last_only=True)
                new_state, new_obs = ps[:, -1].float(), po[:, -1].float()
                # Smooth survival weighting avoids brittle early hard terminal predictions.
                if getattr(args, "predict_termination", True):
                    weights *= 1 - pt[:, -1, 0].float().sigmoid()
                if getattr(args, "predict_availability", True):
                    avail = _legal_mask(pa[:, -1].float())
                # The defender remembers the action it issued in the induced LPA MDP.
                previous_issued = F.one_hot(normal_actions, args.n_actions).float()
                q, hidden = mac.forward_step(new_obs, previous_issued, hidden)
                q = q.float()
                normal_actions = q.masked_fill(~avail, -torch.inf).argmax(-1)
                attacked = attack_actions(mixer, args, q, new_state, normal_actions, avail, followup_mask)
                state_hist = torch.cat((state_hist[:, 1:], new_state[:, None]), 1)
                obs_hist = torch.cat((obs_hist[:, 1:], new_obs[:, None]), 1)
                action_hist = torch.cat((action_hist[:, 1:], torch.zeros_like(action_hist[:, :1])), 1)
                times = torch.cat((times[:, 1:], times[:, -1:] + 1), 1)
                history_valid = torch.cat((history_valid[:, 1:], torch.ones_like(history_valid[:, :1])), 1)
        output[active_rows] = cumulative
        return output.reshape(b, k).detach(), endpoint_valid
