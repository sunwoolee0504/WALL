import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch as th
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from components.action_selectors import EpsilonGreedyActionSelector
from components.wolfpack import attack_actions, joint_q, select_followup
from modules.mixers.dmaq_general import DMAQer
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer


def arguments(**overrides):
    options = dict(n_agents=3, n_actions=4, state_shape=5, mixer="qmix",
                   mixing_embed_dim=8, hypernet_layers=1, hypernet_embed=8,
                   num_kernel=2, adv_hypernet_embed=8, adv_hypernet_layers=1,
                   weighted_head=True, is_minus_one=True, lr=0.01,
                   num_followup_agents=2, followup_selection="q_gradient",
                   epsilon_start=1., epsilon_finish=.1, epsilon_anneal_time=100)
    options.update(overrides)
    return NS(**options)


class AttackTests(unittest.TestCase):
    def setUp(self):
        th.manual_seed(13)
        th.set_num_threads(1)

    def test_training_explores_and_evaluation_is_greedy(self):
        args = arguments()
        selector = EpsilonGreedyActionSelector(args)
        q = th.zeros(4096, 3, 4)
        q[..., 2] = 100
        available = th.ones_like(q)
        available[..., 0] = 0
        train, *_ = selector.select_action_wolfpack(q, available, 0)
        self.assertGreater((train != 2).float().mean(), .5)
        self.assertTrue((train != 0).all())
        self.assertEqual(selector.epsilon, 1.)
        test = selector.select_action(q, available, 0, test_mode=True)
        self.assertTrue((test == 2).all())
        self.assertEqual(selector.epsilon, 0.)

    def test_all_mixers_match_bruteforce_unilateral_attack(self):
        for name, cls in [("vdn", VDNMixer), ("qmix", QMixer), ("dmaq", DMAQer)]:
            args = arguments(mixer=name)
            mixer = cls() if name == "vdn" else cls(args)
            q = th.randn(2, 3, 4)
            state = th.randn(2, 5)
            avail = th.ones_like(q)
            avail[:, 1, 3] = 0
            avail[0, 2] = th.tensor([1, 0, 0, 0])
            actions = q.masked_fill(avail == 0, -th.inf).argmax(-1)
            mask = th.tensor([[True, False, True], [True, True, False]])
            attacked = attack_actions(mixer, args, q, state, actions, avail, mask)
            for batch in range(2):
                for agent in range(3):
                    if not mask[batch, agent] or avail[batch, agent].sum() <= 1:
                        self.assertEqual(attacked[batch, agent], actions[batch, agent])
                        continue
                    scores = []
                    for candidate in range(4):
                        proposed = actions[batch:batch+1].clone()
                        proposed[:, agent] = candidate
                        scores.append(joint_q(mixer, args, q[batch:batch+1], state[batch:batch+1],
                                              proposed, avail[batch:batch+1]).item()
                                      if avail[batch, agent, candidate] else float("inf"))
                    self.assertEqual(int(attacked[batch, agent]), min(range(4), key=scores.__getitem__))

    def test_qplex_candidates_hold_other_actions_at_original(self):
        class TableMixer(th.nn.Module):
            def forward(self, qs, states, actions=None, is_v=False, **kw):
                if is_v:
                    return qs.sum(-1, keepdim=True) * 0
                matrix = qs.new_tensor([[0, 20, -5], [10, 10, 5], [-1, -6, 4]])
                ids = actions.argmax(-1).reshape(-1, 2)
                return matrix[ids[:, 0], ids[:, 1]].reshape(-1, 1, 1)
        args = arguments(mixer="dmaq", n_agents=2, n_actions=3)
        q = th.tensor([[[0., 1., 2.], [0., 1., 2.]]])
        got = attack_actions(TableMixer(), args, q, th.zeros(1, 5), th.tensor([[2, 2]]),
                             th.ones_like(q), th.ones(1, 2, dtype=th.bool))
        self.assertEqual(got.tolist(), [[0, 1]])

    def test_qplex_value_and_advantage_gradient_matches_finite_difference(self):
        args = arguments(mixer="dmaq")
        mixer = DMAQer(args).double()
        q = th.randn(1, 3, 4, dtype=th.double, requires_grad=True)
        state, avail = th.randn(1, 5, dtype=th.double), th.ones_like(q)
        actions = th.tensor([[0, 1, 2]])
        chosen = q.gather(-1, actions.unsqueeze(-1)).squeeze(-1).unsqueeze(1)
        expected = mixer(chosen, state, is_v=True) + mixer(
            chosen, state, actions=F.one_hot(actions, 4).double().unsqueeze(1),
            max_q_i=q.amax(-1).unsqueeze(1), is_v=False)
        actual = joint_q(mixer, args, q, state, actions, avail)
        self.assertTrue(th.allclose(actual, expected.reshape(-1)))
        gradient, = th.autograd.grad(actual.sum(), q)
        epsilon = 1e-5
        for agent in range(3):
            for action in range(4):
                plus, minus = q.detach().clone(), q.detach().clone()
                plus[0, agent, action] += epsilon
                minus[0, agent, action] -= epsilon
                difference = (joint_q(mixer, args, plus, state, actions, avail) -
                              joint_q(mixer, args, minus, state, actions, avail)) / (2 * epsilon)
                self.assertAlmostEqual(gradient[0, agent, action].item(), difference.item(), places=6)

    def test_followup_is_finite_excludes_initial_dead_and_does_not_mutate_parameters(self):
        args = arguments(mixer="dmaq")
        mixer = DMAQer(args)
        before = {key: value.clone() for key, value in mixer.state_dict().items()}
        q = th.zeros(2, 3, 4)
        state = th.randn(2, 5)
        avail = th.ones_like(q)
        avail[:, 2, 1:] = 0
        actions = th.zeros(2, 3, dtype=th.long)
        attacked = actions.clone()
        attacked[:, 0] = 1
        with th.no_grad():
            group = select_followup(mixer, args, q, state, actions, attacked, avail, th.tensor([0, 0]))
        self.assertEqual(group.tolist(), [[1, -1], [1, -1]])
        self.assertTrue(all(parameter.grad is None for parameter in mixer.parameters()))
        self.assertTrue(all(th.equal(before[key], value) for key, value in mixer.state_dict().items()))

    def test_q_gradient_ranking_matches_independent_raw_q_kl(self):
        args = arguments(mixer="vdn", num_followup_agents=1, lr=0.2)
        q = th.tensor([[[2., 0., -1., -2.], [0., 0., 0., 0.], [5., 0., 0., 0.]]])
        actions = th.zeros(1, 3, dtype=th.long)
        attacked = th.tensor([[1, 0, 0]])
        avail, state = th.ones_like(q), th.zeros(1, 5)
        group = select_followup(VDNMixer(), args, q, state, actions, attacked, avail, th.tensor([0]))
        adjusted = q.double().clone()
        adjusted.scatter_add_(-1, attacked.unsqueeze(-1), th.full((1, 3, 1), args.lr, dtype=th.double))
        p = q.double().softmax(-1)
        kl = (p * (q.double().log_softmax(-1) - adjusted.log_softmax(-1))).sum(-1)
        self.assertEqual(group.item(), 1 + kl[:, 1:].argmax(-1).item())


if __name__ == "__main__":
    unittest.main()


def test_omitting_detached_baseline_preserves_followup_for_every_mixer():
    import pytest
    import components.wolfpack as module
    from unittest.mock import patch
    th.manual_seed(921)
    for name, cls in [("vdn", VDNMixer), ("qmix", QMixer), ("dmaq", DMAQer)]:
        args = arguments(mixer=name)
        mixer = cls() if name == "vdn" else cls(args)
        q = th.randn(7,3,4)
        states = th.randn(7,5)
        avail = th.ones_like(q)
        actions = q.argmax(-1)
        attacked = actions.clone(); attacked[:,0] = q[:,0].argmin(-1)
        leaf = q.clone().requires_grad_(True)
        baseline = joint_q(mixer,args,leaf,states,actions,avail).detach()
        grad, = th.autograd.grad((baseline-joint_q(mixer,args,leaf,states,attacked,avail)).sum(),leaf)
        updated = q-args.lr*grad
        before, after = q.double().log_softmax(-1), updated.double().log_softmax(-1)
        score = (before.exp()*(before-after)).sum(-1).clamp_min(0)
        score[:,0] = -th.inf
        expected = score.argsort(dim=-1,descending=True,stable=True)[:,:2]
        with patch.object(module,'joint_q',wraps=joint_q) as wrapped:
            actual = select_followup(mixer,args,q,states,actions,attacked,avail,th.zeros(7,dtype=th.long))
            assert wrapped.call_count == 1
        assert th.equal(actual,expected)
