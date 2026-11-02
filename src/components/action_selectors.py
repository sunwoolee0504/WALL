import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}


def _legal(avail_actions):
    legal = avail_actions.bool().clone()
    legal[..., 0] |= ~legal.any(dim=-1)
    return legal


class EpsilonGreedyActionSelector:
    def __init__(self, args):
        self.args = args
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish,
                                              args.epsilon_anneal_time, decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        self.epsilon = 0.0 if test_mode else self.schedule.eval(t_env)
        legal = _legal(avail_actions)
        greedy = agent_inputs.masked_fill(~legal, -th.inf).argmax(dim=-1)
        if self.epsilon == 0.0:
            return greedy
        random_actions = Categorical(probs=legal.float()).sample().long()
        pick_random = th.rand_like(agent_inputs[..., 0]) < self.epsilon
        return th.where(pick_random, random_actions, greedy)

    def select_action_wolfpack(self, agent_inputs, avail_actions, t_env, test_mode=False):
        """Compatibility tuple; the runner uses full joint-Q attack helpers."""
        actions = self.select_action(agent_inputs, avail_actions, t_env, test_mode)
        legal = _legal(avail_actions)
        random_actions = Categorical(probs=legal.float()).sample().long()
        worst = agent_inputs.masked_fill(~legal, th.inf).argmin(dim=-1)
        return actions, actions.clone(), random_actions, worst


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
