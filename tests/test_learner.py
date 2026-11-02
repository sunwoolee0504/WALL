"""Integration and attack-target regressions on synthetic episodes (no SC2)."""
import copy
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.basic_controller import BasicMAC
from learners.q_learner import QLearner
from learners.dmaq_qatten_learner import DMAQ_qattenLearner
from learners.wall_auxiliary import WALLAuxiliaryLearner, horizon_targets, transition_mask
from modules.step.models import Planning_Transformer, Qdifference_Transformer
from modules.mixers.vdn import VDNMixer


def config(mixer="qmix", **overrides):
    data = dict(n_agents=3, n_actions=4, state_shape=7, obs_shape=5, rnn_hidden_dim=12,
                trans_embed_dim=16, num_trans_head=2, num_trans_layer=1, trans_input_len=5,
                trans_dropout=0.0, trans_ff_multiplier=2, attention_backend="sdpa", attack_period=3,
                attack_duration=2, num_followup_agents=1, followup_selection="q_gradient",
                followup_q_lr=0.01, agent="rnn", agent_output_type="q", obs_last_action=True,
                obs_agent_id=True, action_selector="epsilon_greedy", epsilon_start=1.0,
                epsilon_finish=0.05, epsilon_anneal_time=50000, lr=0.0005,
                optim_alpha=0.99, optim_eps=1e-5, gamma=0.99, grad_norm_clip=10.0,
                double_q=True, mixer=mixer, mixing_embed_dim=8, hypernet_layers=2,
                hypernet_embed=16, adv_hypernet_layers=1, adv_hypernet_embed=16, num_kernel=2,
                is_minus_one=True, weighted_head=True, is_adv_attention=True, is_stop_gradient=True,
                target_update_mode="hard", target_update_interval=200, learner_log_interval=1,
                planner_origins_per_episode=1, planner_train_span=None, planner_batch_size=2,
                world_model_aux_weight=0.1, predict_availability=True, predict_termination=True,
                env_max_timestep=30, episode_limit=30, device="cpu", use_amp=False, compile_models=False)
    data.update(overrides)
    return SimpleNamespace(**data)


def make_batch(args, lengths=(7, 3), padded_length=None):
    length = padded_length or max(lengths) + 1
    scheme = {"state": {"vshape": args.state_shape},
              "obs": {"vshape": args.obs_shape, "group": "agents"},
              "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
              "forced_actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
              "avail_actions": {"vshape": (args.n_actions,), "group": "agents", "dtype": torch.int},
              "reward": {"vshape": (1,)}, "terminated": {"vshape": (1,), "dtype": torch.uint8}}
    preprocess = {"actions": ("actions_onehot", [OneHot(args.n_actions)]),
                  "forced_actions": ("forced_actions_onehot", [OneHot(args.n_actions)])}
    batch = EpisodeBatch(scheme, {"agents": args.n_agents}, len(lengths), length,
                         preprocess=preprocess, device="cpu")
    for i, steps in enumerate(lengths):
        actions = torch.randint(args.n_actions, (1, steps + 1, args.n_agents, 1))
        executed = actions.clone()
        executed[..., 0, :] = (executed[..., 0, :] + 1) % args.n_actions
        terminal = torch.zeros(1, steps + 1, 1, dtype=torch.uint8)
        if steps:
            terminal[:, steps - 1] = 1
        batch.update({"state": torch.randn(1, steps + 1, args.state_shape),
                      "obs": torch.randn(1, steps + 1, args.n_agents, args.obs_shape),
                      "actions": actions, "forced_actions": executed,
                      "avail_actions": torch.ones(1, steps + 1, args.n_agents, args.n_actions, dtype=torch.int),
                      "reward": torch.randn(1, steps + 1, 1), "terminated": terminal},
                     bs=slice(i, i + 1), ts=slice(0, steps + 1))
    return batch


def make_learner(args, batch):
    mac = BasicMAC(batch.scheme, {"agents": args.n_agents}, args)
    cls = DMAQ_qattenLearner if args.mixer == "dmaq" else QLearner
    return cls(mac, batch.scheme, None, args, Qdifference_Transformer(args), Planning_Transformer(args)).to("cpu")


@pytest.mark.parametrize("mixer", ["vdn", "qmix", "dmaq"])
@pytest.mark.parametrize("lengths", [(7, 3), (1, 2)])
def test_complete_update_is_finite_and_updates_all_models(mixer, lengths):
    torch.manual_seed(8)
    args = config(mixer)
    batch = make_batch(args, lengths)
    learner = make_learner(args, batch)
    before = [next(m.parameters()).detach().clone() for m in
              (learner.mac.agent, learner.planning_transformer, learner.qdiff_transformer)]
    result = learner.train(batch, t_env=1, episode_num=1)
    assert learner.wall_updates == 1
    assert all(np.isfinite(value) for value in result.values())
    for old, model in zip(before, (learner.mac.agent, learner.planning_transformer, learner.qdiff_transformer)):
        assert not torch.equal(old, next(model.parameters()))
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad is None for p in learner.target_mac.parameters())
    assert all(p.grad is None for p in learner.target_mixer.parameters())


def test_future_targets_exclude_terminal_padding_and_never_wrap():
    effects = torch.tensor([[[1., 2., 900., 999.]]])
    valid = torch.tensor([[[True, True, False, False]]])
    target, mask = horizon_targets(effects, valid, horizon=5)
    assert mask.sum().item() == 3
    assert torch.equal(target[mask], torch.tensor([1., 2., 2.]))
    pred = target.clone().requires_grad_()
    loss = ((pred - target).square() * mask).sum()
    loss.backward()
    assert pred.grad[~mask].count_nonzero() == 0


def test_transition_mask_includes_terminal_transition_only():
    args = config()
    batch = make_batch(args, (1, 2), padded_length=5)
    assert torch.equal(transition_mask(batch), torch.tensor([[True, False, False, False], [True, True, False, False]]))


@pytest.mark.parametrize("mixer", ["vdn", "qmix", "dmaq"])
def test_checkpoint_round_trip_and_reproducible_next_update(tmp_path, mixer):
    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)
    args = config(mixer)
    batch = make_batch(args)
    learner = make_learner(args, batch)
    learner.train(batch, 1, 1)
    learner.save_models(tmp_path)
    first_rng = (random.random(), float(np.random.rand()), torch.rand(3))
    learner.train(batch, 2, 2)
    reference = {k: v.detach().clone() for k, v in learner.mac.agent.state_dict().items()}
    other = make_learner(args, batch)
    other.load_models(tmp_path, resume=True)
    second_rng = (random.random(), float(np.random.rand()), torch.rand(3))
    assert first_rng[:2] == second_rng[:2]
    torch.testing.assert_close(first_rng[2], second_rng[2], rtol=0, atol=0)
    other.train(batch, 2, 2)
    for k, value in other.mac.agent.state_dict().items():
        torch.testing.assert_close(value, reference[k], rtol=0, atol=0)
    assert other.wall_updates == 2
    for name in ("planning_transformer", "qdiff_transformer", "mixer", "target_mixer"):
        for a, b in zip(getattr(learner, name).parameters(), getattr(other, name).parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_pretrain_load_synchronizes_target_mixer(tmp_path):
    args = config()
    batch = make_batch(args)
    learner = make_learner(args, batch)
    learner.mac.save_models(str(tmp_path))
    values = {key: torch.full_like(value, 0.25) for key, value in learner.mixer.state_dict().items()}
    torch.save(values, tmp_path / "mixer.th")
    learner.load_models(tmp_path)
    for key, value in learner.mixer.state_dict().items():
        torch.testing.assert_close(value, learner.target_mixer.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="Full resume"):
        learner.load_models(tmp_path, resume=True)


def test_paper_effect_uses_same_state_gap_and_advances_hidden():
    args = config("vdn", n_agents=2, n_actions=2, obs_shape=1, state_shape=1, rnn_hidden_dim=1,
                  predict_availability=False, predict_termination=False, num_followup_agents=1)
    batch = make_batch(args, (1,))

    class Planner(nn.Module):
        def forward(self, state, obs, actions, **kwargs):
            # Any difference between separate baseline/attack state trajectories
            # would add unrelated Q growth. The expected gap is exactly 2+3+4.
            return state + 10, obs + 1, torch.zeros_like(actions), torch.zeros_like(state)

    class MAC:
        def forward_step(self, obs, previous, hidden):
            assert torch.equal(previous[..., 0], torch.ones_like(previous[..., 0]))
            new = hidden + 1
            return torch.cat((new + 1, torch.zeros_like(new)), -1), new

    owner = SimpleNamespace(args=args, planning_transformer=Planner(), qdiff_transformer=nn.Identity(),
                            mac=MAC(), mixer=VDNMixer())
    auxiliary = WALLAuxiliaryLearner(owner)
    # Disable autocast helper device inspection for this parameter-free oracle.
    from contextlib import nullcontext
    auxiliary.autocast = nullcontext
    q = torch.tensor([[[[2., 0.], [2., 0.]], [[0., 0.], [0., 0.]]]])
    hidden = torch.ones(1, 2, 2, 1)
    effects, valid = auxiliary.attack_effects(batch, q, hidden, transition_mask(batch), torch.tensor([[0]]))
    torch.testing.assert_close(effects, torch.tensor([[9.]]))
    assert valid.all()


def test_noop_targets_skip_planner_and_followup(monkeypatch):
    args = config("vdn")
    batch = make_batch(args, (2,))
    learner = make_learner(args, batch)
    q, h = learner.mac.forward_sequence(batch, return_hidden=True)
    # Only the same single legal action exists for every agent.
    batch.data.transition_data["avail_actions"].zero_()
    batch.data.transition_data["avail_actions"][..., 0] = 1
    from unittest.mock import Mock
    planner = Mock(side_effect=AssertionError("No-op initial attack must not simulate"))
    monkeypatch.setattr(learner.auxiliary, "_planner_call", planner)
    import learners.wall_auxiliary as module
    monkeypatch.setattr(module, "select_followup", Mock(side_effect=AssertionError("No-op group computation")))
    effects, valid = learner.auxiliary.attack_effects(batch, q.detach(), h.detach(), transition_mask(batch), torch.tensor([[0, 1, 5]]))
    assert effects.count_nonzero() == 0
    assert torch.equal(valid, torch.tensor([[True, True, False]]))


@pytest.mark.parametrize("mixer", ["qmix", "dmaq"])
def test_effect_construction_preserves_existing_policy_and_mixer_gradients(mixer):
    args = config(mixer)
    batch = make_batch(args)
    learner = make_learner(args, batch)
    q, h = learner.mac.forward_sequence(batch, return_hidden=True)
    params = list(learner.mac.parameters()) + list(learner.mixer.parameters())
    for param in params:
        param.grad = torch.randn_like(param)
    grads = [p.grad.clone() for p in params]
    weights = [p.detach().clone() for p in params]
    learner.auxiliary.attack_effects(batch, q.detach(), h.detach(), transition_mask(batch), torch.tensor([[0, 3], [0, 2]]))
    for p, g, w in zip(params, grads, weights):
        torch.testing.assert_close(p.grad, g, rtol=0, atol=0)
        torch.testing.assert_close(p, w, rtol=0, atol=0)


def test_context_and_horizon_can_differ_but_all_logits_are_supervised():
    args = config(trans_input_len=3, attack_period=7, planner_train_span=None)
    batch = make_batch(args, (10, 8))
    learner = make_learner(args, batch)
    result = learner.train(batch, 1, 1)
    assert result["qdiff_valid_pairs"] > 0
    # Each horizon output participates, including horizons beyond context=3.
    gradient = learner.qdiff_transformer.predict_qdiff.weight.grad
    assert (gradient.abs().sum(-1) > 0).all()
    args.planner_train_span = 3
    with pytest.raises(ValueError, match="every attack_period"):
        learner.train(batch, 2, 2)


def test_ema_targets_move_by_configured_tau():
    args = config(target_update_mode="ema", target_update_tau=0.1)
    batch = make_batch(args)
    learner = make_learner(args, batch)
    old = [p.detach().clone() for p in learner.target_mac.parameters()]
    learner.train(batch, 1, 1)
    for previous, online, target in zip(old, learner.mac.parameters(), learner.target_mac.parameters()):
        torch.testing.assert_close(target, previous.lerp(online, 0.1), rtol=1e-6, atol=1e-7)


def test_resume_rejects_different_map_even_with_identical_dimensions(tmp_path):
    args = config(env_args={"map_name": "map_a"})
    batch = make_batch(args)
    learner = make_learner(args, batch)
    learner.save_models(tmp_path)
    args.env_args["map_name"] = "map_b"
    with pytest.raises(ValueError, match="architecture/map"):
        learner.load_models(tmp_path, resume=True)


def test_warm_start_retains_moments_but_honors_configured_optimizer_settings(tmp_path):
    original_args = config()
    batch = make_batch(original_args)
    original = make_learner(original_args, batch)
    original.train(batch, 1, 1)
    original.save_models(tmp_path)
    args = config(lr=1e-4, optim_alpha=0.95, optim_eps=2e-5,
                  planner_lr=3e-4, qdiff_lr=4e-4)
    other = make_learner(args, batch)
    other.load_models(tmp_path, resume=False)
    group = other.optimiser.param_groups[0]
    assert (group["lr"], group["alpha"], group["eps"]) == (1e-4, 0.95, 2e-5)
    assert other.optimizer_metrics() == dict(lr_defender=1e-4, lr_planner=3e-4, lr_qdiff=4e-4)
    saved = original.optimiser.state_dict()["state"]
    loaded = other.optimiser.state_dict()["state"]
    assert saved and saved.keys() == loaded.keys()
    for parameter, state in saved.items():
        for key, value in state.items():
            torch.testing.assert_close(loaded[parameter][key], value, rtol=0, atol=0)
    result = other.train(batch, 2, 2)
    assert result["lr_defender"] == 1e-4
    assert all(np.isfinite(value) for value in result.values())
    # Full resume is a different contract: restore the old run, including rates.
    other.load_models(tmp_path, resume=True)
    assert other.optimizer_metrics() == original.optimizer_metrics()
    assert other.optimiser.param_groups[0]["alpha"] == original_args.optim_alpha


def test_auxiliary_rates_default_to_defender_rate():
    args = config(lr=2.5e-4, planner_lr=None, qdiff_lr=None)
    learner = make_learner(args, make_batch(args))
    assert set(learner.optimizer_metrics().values()) == {2.5e-4}
