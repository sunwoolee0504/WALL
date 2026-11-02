import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from components.action_selectors import EpsilonGreedyActionSelector
from components.transforms import OneHot
from controllers.basic_controller import BasicMAC
from modules.mixers.qmix import QMixer
from modules.mixers.dmaq_general import DMAQer
from envs import REGISTRY
from modules.mixers.vdn import VDNMixer
from runners.episode_runner import EpisodeRunner


class TinyEnv:
    def __init__(self, episode_limit=9, dead_at=None):
        self.episode_limit = episode_limit
        self.dead_at = dead_at
        self.executed = []
        self.time = 0

    def reset(self):
        self.time = 0
        self.executed = []

    def get_state(self):
        return np.array([self.time, 0.], dtype=np.float32)

    def get_obs(self):
        return np.full((3, 2), self.time, dtype=np.float32)

    def get_avail_actions(self):
        available = np.ones((3, 2), dtype=np.int32)
        if self.time == self.dead_at:
            available[1, 1] = 0
        return available

    def step(self, actions):
        self.executed.append(np.array(actions))
        self.time += 1
        return 1., self.time == self.episode_limit, {"battle_won": 1}


class FakeMAC:
    def __init__(self, args):
        self.agent = th.nn.Linear(1, 1)
        self.action_selector = EpsilonGreedyActionSelector(args)
        self.calls = []

    def init_hidden(self, batch_size):
        self.calls = []

    def forward(self, batch, t, test_mode=False):
        self.calls.append((t, th.is_grad_enabled()))
        return th.tensor([[[1., -1.]] * 3])


class SpyPredictor(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.times = None

    def forward(self, times, states, last_only=False):
        self.times = times.clone()
        return th.tensor([0., 1., 2., 3.]).expand(states.shape[0], states.shape[1], -1)


def make_runner(**changes):
    args = dict(batch_size_run=1, env="tiny_wall_test", env_args={}, device="cpu",
                n_agents=3, n_actions=2, num_attack_train=8, num_attack_test=8,
                attack_duration=3, attack_period=4, trans_input_len=4, temperature=1.,
                init_attack_step_method="planner", init_agent_random=True,
                num_followup_agents=1, followup_selection="q_gradient", lr=.01,
                mixer="vdn", test_nepisode=2, runner_log_interval=10000,
                epsilon_start=0., epsilon_finish=0., epsilon_anneal_time=100,
                planner_warmup_updates=0)
    args.update(changes)
    args = NS(**args)
    REGISTRY["tiny_wall_test"] = TinyEnv
    logger = NS(log_stat=Mock())
    runner = EpisodeRunner(args, logger)
    scheme = {
        "state": {"vshape": 2}, "obs": {"vshape": 2, "group": "agents"},
        "avail_actions": {"vshape": 2, "group": "agents", "dtype": th.int32},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "forced_actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "reward": {"vshape": (1,)}, "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    preprocess = {"actions": ("actions_onehot", [OneHot(2)]),
                  "forced_actions": ("forced_actions_onehot", [OneHot(2)])}
    runner.setup(scheme, {"agents": 3}, preprocess, FakeMAC(args), SpyPredictor(), None)
    runner.setup_learner(NS(mixer=VDNMixer(), wall_updates=100))
    return runner


class RunnerTests(unittest.TestCase):
    def test_3m_train_eight_eval_four_use_two_and_one_attack_bursts(self):
        for test_mode, budget, bursts in ((False, 8, 2), (True, 4, 1)):
            runner = make_runner(num_attack_train=8, num_attack_test=4)
            with patch.object(runner, "get_attack_step", return_value=(True, 0)) as schedule, \
                 patch("runners.episode_runner.select_followup", return_value=th.tensor([[1]])) as choose:
                runner.run_wolfpack_attacker(test_mode=test_mode)
            changed = np.any(np.asarray(runner.env.executed) != 0, axis=1)
            self.assertEqual(changed.tolist(), [True] * budget + [False] * (9 - budget))
            self.assertEqual(schedule.call_count, bursts)
            self.assertEqual(choose.call_count, bursts)

    def test_exact_followup_duration_noop_budget_fixed_groups_and_burst_cap(self):
        runner = make_runner(env_args={"dead_at": 1})
        with patch.object(runner, "get_attack_step", return_value=(True, 0)) as schedule, \
             patch("runners.episode_runner.select_followup", return_value=th.tensor([[1]])) as choose:
            batch = runner.run_wolfpack_attacker(test_mode=True)
        self.assertEqual(schedule.call_count, 2)
        self.assertEqual(choose.call_count, 2)
        expected = [[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0],
                    [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]]
        self.assertEqual(np.asarray(runner.env.executed).tolist(), expected)
        self.assertEqual(runner.test_wolfpack_stats["attack_changed_steps"], 7)
        self.assertEqual(runner.test_wolfpack_stats["attack_bursts"], 2)
        self.assertTrue((batch["actions"] == 0).all())
        self.assertEqual(batch["forced_actions"][0, :9, :, 0].tolist(), expected)
        self.assertEqual(runner.mac.calls, [(time, False) for time in range(10)])

    def test_zero_budget_skips_predictor_and_followup(self):
        runner = make_runner(num_attack_test=0)
        with patch.object(runner, "get_attack_step", side_effect=AssertionError("unneeded predictor")), \
             patch("runners.episode_runner.select_followup", side_effect=AssertionError("unneeded gradient")):
            batch = runner.run_wolfpack_attacker(test_mode=True)
        self.assertTrue(th.equal(batch["actions"], batch["forced_actions"]))

    def test_noop_initial_does_not_start_burst(self):
        runner = make_runner(env_args={"episode_limit": 5})
        # Constant q has the same greedy/min action, hence no actual changes.
        runner.mac.forward = Mock(return_value=th.zeros(1, 3, 2))
        with patch.object(runner, "get_attack_step", return_value=(True, 0)) as schedule, \
             patch("runners.episode_runner.select_followup", side_effect=AssertionError("no burst")):
            runner.run_wolfpack_attacker(test_mode=True)
        self.assertEqual(schedule.call_count, 5)
        self.assertEqual(runner.test_wolfpack_stats["attack_changed_steps"], 0)
        self.assertEqual(runner.test_wolfpack_stats["attack_bursts"], 0)

    def test_attacked_evaluation_aggregates_and_restores_model_modes(self):
        runner = make_runner(num_attack_test=0, env_args={"episode_limit": 3})
        runner.run_wolfpack_attacker(test_mode=True)
        runner.logger.log_stat.assert_not_called()
        self.assertTrue(runner.qdiff_transformer.training)
        self.assertTrue(runner.mac.agent.training)
        runner.run_wolfpack_attacker(test_mode=True)
        runner.logger.log_stat.assert_any_call("test_Wolfpack_return_mean", 3., 0)
        self.assertEqual(runner.test_wolfpack_returns, [])
        self.assertEqual(runner.test_wolfpack_stats, {})

    def test_scheduler_keeps_fixed_horizon_and_correct_absolute_timestamps(self):
        runner = make_runner(env_args={"episode_limit": 100})
        runner.reset()
        runner.t = 21
        with patch("runners.episode_runner.th.rand", return_value=th.tensor(.2)):
            self.assertEqual(runner.get_attack_step(pre_initial_attack_step=0), (False, -1))
            self.assertEqual(runner.get_attack_step(pre_initial_attack_step=21), (False, -1))
        self.assertEqual(runner.qdiff_transformer.times.tolist(), [[18, 19, 20, 21]])

    def test_real_controller_and_all_mixers_complete_attacked_episode(self):
        th.manual_seed(123)
        for name, mixer_class in [("vdn", VDNMixer), ("qmix", QMixer), ("dmaq", DMAQer)]:
            runner = make_runner(mixer=name, env_args={"episode_limit": 5},
                                 agent="rnn", agent_output_type="q", obs_last_action=True,
                                 obs_agent_id=True, rnn_hidden_dim=8, action_selector="epsilon_greedy",
                                 state_shape=2, mixing_embed_dim=4, hypernet_layers=1,
                                 hypernet_embed=8, num_kernel=2, adv_hypernet_embed=8,
                                 weighted_head=True, is_minus_one=True)
            sample = runner.new_batch()
            runner.mac = BasicMAC(sample.scheme, {"agents": 3}, runner.args)
            runner.learner.mixer = mixer_class() if name == "vdn" else mixer_class(runner.args)
            with patch.object(runner, "get_attack_step", return_value=(True, 0)):
                batch = runner.run_wolfpack_attacker(test_mode=True)
            self.assertEqual(len(runner.env.executed), 5)
            self.assertTrue(th.isfinite(runner.mac.hidden_states).all())
            self.assertLessEqual(runner.test_wolfpack_stats["attack_changed_steps"], 8)
            self.assertTrue(all(parameter.grad is None for parameter in runner.mac.parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in runner.learner.mixer.parameters()))
            self.assertFalse(batch["obs"].requires_grad)

    def test_natural_collection_records_executed_actions_for_dynamics(self):
        runner = make_runner(env_args={"episode_limit": 3})
        batch = runner.run(test_mode=False)
        self.assertTrue(th.equal(batch["actions"], batch["forced_actions"]))
        self.assertEqual(runner.t_env, 3)


if __name__ == "__main__":
    unittest.main()
