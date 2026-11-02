"""Runtime checks require neither SMAC, CUDA nor any external telemetry."""
from copy import deepcopy
import logging
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from components.episode_buffer import EpisodeBatch, ReplayBuffer
from components.transforms import OneHot
from run import WALL_MAP_PRESETS, _checkpoint_directory, _evaluate, args_sanity_check, isolated_rng, seed_everything
from utils.logging import Logger


ROOT = Path(__file__).resolve().parents[1]


def config():
    result = yaml.safe_load((ROOT / "src/config/default.yaml").read_text())
    result.update(yaml.safe_load((ROOT / "src/config/algs/wall_qmix.yaml").read_text()))
    result["env_args"] = {"map_name": "3m"}
    return result


@pytest.mark.parametrize("map_name,preset", list(WALL_MAP_PRESETS.items()))
def test_map_presets_and_cli_overrides(map_name, preset):
    cfg = config()
    cfg["env_args"]["map_name"] = map_name
    fixed = args_sanity_check(cfg, logging.getLogger(__name__))
    assert tuple(fixed[k] for k in ("num_attack_train", "num_attack_test", "num_followup_agents")) == preset
    assert cfg["num_attack_train"] is None
    cfg["num_attack_train"] = 23
    assert args_sanity_check(cfg, logging.getLogger(__name__))["num_attack_train"] == 23


@pytest.mark.parametrize("changes", [{"runner": "parallel"}, {"batch_size_run": 2},
    {"attack_period": 0}, {"temperature": 0}, {"trans_embed_dim": 63, "num_trans_head": 2},
    {"buffer_size": 2}, {"num_attack_train": -1}, {"init_agent_random": False}])
def test_invalid_configs_fail_before_environment(changes):
    cfg = config()
    cfg.update(changes)
    with pytest.raises(ValueError):
        args_sanity_check(cfg, logging.getLogger(__name__))


def test_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = config()
    cfg.update(use_cuda=True, use_amp=True)
    fixed = args_sanity_check(cfg, logging.getLogger(__name__))
    assert not fixed["use_cuda"] and not fixed["use_amp"]


def random_values():
    return random.random(), np.random.random(), torch.rand(3)


def assert_random_values_equal(left, right):
    assert left[:2] == right[:2]
    assert torch.equal(left[2], right[2])


def test_all_rngs_seeded_and_evaluation_restored_even_on_error():
    seed_everything(431)
    expected = random_values()
    seed_everything(431)
    with pytest.raises(RuntimeError):
        with isolated_rng(19):
            random_values()
            raise RuntimeError("deliberate evaluation failure")
    assert_random_values_equal(expected, random_values())


def test_evaluation_runs_both_conditions_and_keeps_env_open():
    events = []
    runner = SimpleNamespace(run=lambda **kw: events.append(("natural", kw)),
                             run_wolfpack_attacker=lambda **kw: events.append(("attacked", kw)),
                             save_replay=lambda: events.append(("save", {})))
    _evaluate(SimpleNamespace(test_nepisode=3, save_replay=True), runner, 78)
    assert [name for name, _ in events] == ["natural", "attacked"] * 3 + ["save"]


class CountingOneHot(OneHot):
    count = 0

    def transform(self, tensor):
        self.count += 1
        return super().transform(tensor)


def batch_scheme():
    return {"state": {"vshape": 2}, "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long}}


def test_buffer_copies_without_graph_and_skips_duplicate_onehot():
    transform = CountingOneHot(3)
    preprocess = {"actions": ("actions_onehot", [transform])}
    batch = EpisodeBatch(batch_scheme(), {"agents": 2}, 1, 4, preprocess=preprocess)
    values = torch.tensor([[1.0, 2.0]], requires_grad=True)
    batch.update({"state": values, "actions": np.array([[1, 2]])}, ts=0)
    assert transform.count == 1
    assert not batch["state"].requires_grad
    values.detach().fill_(7)
    assert batch["state"][0, 0].tolist() == [1, 2]
    replay = ReplayBuffer(batch_scheme(), {"agents": 2}, 3, 4, preprocess=preprocess)
    replay.insert_episode_batch(batch)
    assert transform.count == 1
    assert torch.equal(replay["actions_onehot"][0], batch["actions_onehot"][0])
    assert replay.max_t_filled() == 1
    sample = replay.sample(1)
    sample.to("cpu", non_blocking=True)
    assert sample.device == "cpu"
    with pytest.raises(ValueError):
        batch.update({"state": torch.zeros(2, 2)}, ts=0)


def test_replay_wrap_and_short_episode_clear_stale_tail():
    scheme = {"state": {"vshape": 2}}
    replay = ReplayBuffer(scheme, {}, 2, 5)
    long = EpisodeBatch(scheme, {}, 2, 5)
    long.update({"state": np.ones((2, 5, 2))})
    replay.insert_episode_batch(long)
    assert replay.buffer_index == 0 and replay.episodes_in_buffer == 2
    short = EpisodeBatch(scheme, {}, 1, 2)
    short.update({"state": np.full((1, 2, 2), 3)})
    replay.insert_episode_batch(short)
    assert replay["filled"][0, 2:].sum() == 0
    assert replay["state"][0, 2:].sum() == 0
    assert replay.sample(2).batch_size == 2
    assert replay[torch.tensor([0])].batch_size == 1


def test_offline_scalar_logger_and_bounded_console_history():
    logger = Logger(logging.getLogger(__name__))
    logger.setup_sacred(SimpleNamespace(info={}))
    for t in range(10):
        logger.log_stat("loss", torch.tensor(float(t)), t)
    assert logger.wandb_run is None
    assert len(logger.stats["loss"]) == 5
    assert logger.sacred_info["loss"] == list(range(10))
    logger.log_stat("episode", np.int64(2), 10)
    logger.print_recent_stats()
    logger.close()


def test_checkpoint_path_has_useful_errors_and_nearest_step(tmp_path):
    with pytest.raises(FileNotFoundError, match="No checkpoint"):
        _checkpoint_directory(tmp_path)
    for step in (10, 30):
        (tmp_path / str(step)).mkdir()
    assert _checkpoint_directory(tmp_path)[1] == 30
    assert _checkpoint_directory(tmp_path, 12)[1] == 10
    (tmp_path / "learner.th").touch()
    assert _checkpoint_directory(tmp_path)[0] == tmp_path


def test_main_modern_mapping_and_yaml_loader():
    from main import _get_config, recursive_dict_update
    assert recursive_dict_update({"a": {"b": 1}}, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}
    assert recursive_dict_update({"a": 1}, None) == {"a": 1}
    params = ["prog", "--config=wall_qmix"]
    assert _get_config(params, "--config", "algs")["mixer"] == "qmix"
    assert params == ["prog"]


def test_ema_alias_and_separate_context_horizon_span():
    cfg = config()
    cfg.update(target_update_mode="soft", trans_input_len=3, attack_period=8)
    fixed = args_sanity_check(cfg, logging.getLogger(__name__))
    assert fixed["target_update_mode"] == "ema"
    assert fixed["planner_train_span"] == 8
    cfg["planner_train_span"] = 7
    with pytest.raises(ValueError, match="at least attack_period"):
        args_sanity_check(cfg, logging.getLogger(__name__))


def test_training_runtime_checkpoint_replay_and_separate_eval_environment(tmp_path, monkeypatch):
    from envs import REGISTRY
    from run import run_sequential

    environments = []

    class RuntimeEnv:
        episode_limit = 3

        def __init__(self, seed, **kwargs):
            self.seed = seed
            self.closed = False
            self.steps = 0
            environments.append(self)

        def get_env_info(self):
            return dict(n_agents=3, n_actions=3, state_shape=4, obs_shape=3, episode_limit=3)

        def reset(self):
            self.steps = 0

        def get_state(self):
            return np.full(4, self.steps / 3, np.float32)

        def get_obs(self):
            return np.full((3, 3), self.steps / 3, np.float32)

        def get_avail_actions(self):
            return np.ones((3, 3), np.int32)

        def step(self, actions):
            self.steps += 1
            return 0.2, self.steps == 3, {"battle_won": 0}

        def close(self):
            self.closed = True

    monkeypatch.setitem(REGISTRY, "sc2", RuntimeEnv)
    cfg = config()
    cfg.update(env="sc2", use_cuda=False, device="cpu", pretrain=False,
               batch_size=1, buffer_size=2, t_max=6, test_nepisode=1,
               test_interval=1000, save_model=True, save_model_interval=1000,
               checkpoint_save_replay=True, local_results_path=str(tmp_path), unique_token="initial",
               rnn_hidden_dim=8, trans_embed_dim=8, trans_ff_multiplier=2,
               trans_input_len=3, attack_period=3, mixing_embed_dim=8, hypernet_embed=8)
    cfg = args_sanity_check(cfg, logging.getLogger(__name__))
    seed_everything(cfg["seed"])
    run_sequential(SimpleNamespace(**cfg), Logger(logging.getLogger(__name__)))
    assert [env.seed for env in environments] == [1, 10001]
    assert all(env.closed for env in environments)
    checkpoint = tmp_path / "models/initial/6"
    runtime = torch.load(checkpoint / "runtime_state.th", weights_only=False)
    assert runtime["episode"] == 2 and runtime["t_env"] == 6
    assert runtime["replay"]["episodes_in_buffer"] == 2
    assert "hidden_states" not in runtime["replay"]["transition_data"]
    cfg.update(checkpoint_path=str(checkpoint), t_max=9, unique_token="resumed")
    run_sequential(SimpleNamespace(**cfg), Logger(logging.getLogger(__name__)))
    resumed = tmp_path / "models/resumed/9"
    runtime = torch.load(resumed / "runtime_state.th", weights_only=False)
    learner = torch.load(resumed / "learner.th", weights_only=False)
    assert runtime["episode"] == 3 and runtime["replay"]["buffer_index"] == 1
    assert learner["wall_updates"] == 3
    assert all(env.closed for env in environments)


@pytest.mark.parametrize("indices", [[0, 2], np.array([0, 2]), torch.tensor([0, 2])])
def test_batch_updates_with_advanced_indices_write_back(indices):
    batch = EpisodeBatch(batch_scheme(), {"agents": 2}, 3, 4,
                         preprocess={"actions": ("actions_onehot", [OneHot(3)])})
    batch.update({"state": torch.tensor([[1., 2.], [3., 4.]]),
                  "actions": torch.tensor([[1, 2], [2, 0]])}, bs=indices, ts=1)
    torch.testing.assert_close(batch["state"][[0, 2], 1], torch.tensor([[1., 2.], [3., 4.]]))
    assert batch["state"][1].count_nonzero() == 0
    assert batch["filled"][:, 1, 0].tolist() == [1, 0, 1]
    assert batch["actions_onehot"][[0, 2], 1].sum().item() == 4


@pytest.mark.parametrize("algorithm", ["wall_qmix", "wall_vdn", "wall_qplex"])
def test_3m_author_corrected_budget_and_explicit_override(algorithm):
    cfg = config()
    cfg.update(yaml.safe_load((ROOT / ("src/config/algs/" + algorithm + ".yaml")).read_text()))
    fixed = args_sanity_check(cfg, logging.getLogger(__name__))
    assert fixed["num_attack_train"] == 8
    assert fixed["num_attack_test"] == 4
    assert fixed["num_followup_agents"] == 1
    assert fixed["attack_duration"] == 3
    assert fixed["pretrain"] is True  # WALL fine-tunes the supplied defender by default.
    cfg.update(num_attack_train=12, num_attack_test=8, pretrain=False)
    fixed = args_sanity_check(cfg, logging.getLogger(__name__))
    assert (fixed["num_attack_train"], fixed["num_attack_test"]) == (12, 8)
    assert fixed["pretrain"] is False


@pytest.mark.parametrize("changes", [
    {"followup_q_lr": float("nan")}, {"trans_dropout": 1}, {"lr": 0},
    {"world_model_aux_weight": -1}, {"attention_backend": "typo"},
    {"epsilon_start": 0.1, "epsilon_finish": 0.2}, {"epsilon_anneal_time": 0},
    {"init_attack_step_method": "typo"}, {"test_greedy": False},
    {"random_attack_probability": 2}, {"learner": "coma_learner"},
])
def test_public_invalid_hyperparameters_fail_early(changes):
    cfg = config()
    cfg.update(changes)
    with pytest.raises(ValueError):
        args_sanity_check(cfg, logging.getLogger(__name__))


@pytest.mark.parametrize("params,key,folder", [
    (["prog"], "--config", "algs"),
    (["prog", "--config=qmix"], "--config", "algs"),
    (["prog", "--config=../../README"], "--config", "algs"),
    (["prog", "--env-config=sc2_beta"], "--env-config", "envs"),
])
def test_public_cli_rejects_missing_or_unsupported_config(params, key, folder):
    from main import _get_config
    with pytest.raises(ValueError, match="must be one of"):
        _get_config(params, key, folder)


@pytest.mark.parametrize("key", ["planner_lr", "qdiff_lr"])
@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "0.0005"])
def test_auxiliary_learning_rates_reject_invalid_values(key, value):
    cfg = config()
    cfg[key] = value
    with pytest.raises(ValueError, match=key):
        args_sanity_check(cfg, logging.getLogger(__name__))


def test_auxiliary_learning_rates_resolve_without_changing_input():
    cfg = config()
    cfg.update(lr=1e-4, planner_lr=5e-4, qdiff_lr=None)
    fixed = args_sanity_check(cfg, logging.getLogger(__name__))
    assert fixed["lr"] == fixed["qdiff_lr"] == 1e-4
    assert fixed["planner_lr"] == 5e-4
    assert cfg["qdiff_lr"] is None
