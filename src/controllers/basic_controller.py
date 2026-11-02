from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


class BasicMAC:
    """Shared policy with equivalent single-step and fused sequence execution."""

    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self._build_agents(self._get_input_shape(scheme))
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None
        self.last_hidden_before = None
        self.last_q_values = None
        self._agent_ids = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        return self.action_selector.select_action(outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

    def select_actions_wolfpack(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        return self.action_selector.select_action_wolfpack(outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

    def _process_outputs(self, outputs, avail_actions, test_mode):
        if self.agent_output_type != "pi_logits":
            return outputs
        mask = getattr(self.args, "mask_before_softmax", True)
        if mask:
            outputs = outputs.masked_fill(avail_actions == 0, -1e10)
        outputs = th.nn.functional.softmax(outputs, dim=-1)
        if not test_mode:
            action_count = avail_actions.sum(dim=-1, keepdim=True).clamp_min(1) if mask else outputs.size(-1)
            outputs = (1 - self.action_selector.epsilon) * outputs + self.action_selector.epsilon / action_count
            if mask:
                outputs = outputs.masked_fill(avail_actions == 0, 0)
        return outputs

    def forward(self, ep_batch, t, test_mode=False):
        self.last_hidden_before = self.hidden_states
        outputs, hidden = self.agent(self._build_inputs(ep_batch, t), self.hidden_states)
        self.hidden_states = hidden.reshape(ep_batch.batch_size, self.n_agents, -1)
        self.last_q_values = outputs.reshape(ep_batch.batch_size, self.n_agents, -1)
        return self._process_outputs(self.last_q_values, ep_batch["avail_actions"][:, t], test_mode)

    def forward_sequence(self, batch, return_hidden=False):
        """Full episodes from zero hidden; no mutation of online recurrent state.

        Returns Q [B,T,N,A], and optionally hidden-after [B,T,N,H]. Previous
        actions are the policy proposals, matching the attacked-MDP TD objective.
        """
        obs = batch["obs"]
        batch_size, length = obs.shape[:2]
        inputs = [obs]
        if self.args.obs_last_action:
            actions = batch["actions_onehot"]
            inputs.append(th.cat((th.zeros_like(actions[:, :1]), actions[:, :-1]), dim=1))
        if self.args.obs_agent_id:
            inputs.append(self._ids(obs).reshape(1, 1, self.n_agents, self.n_agents).expand(batch_size, length, -1, -1))
        inputs = th.cat(inputs, dim=-1).transpose(1, 2).reshape(batch_size * self.n_agents, length, -1)
        outputs, hidden = self.agent.forward_sequence(inputs)
        outputs = outputs.reshape(batch_size, self.n_agents, length, -1).transpose(1, 2)
        hidden = hidden.reshape(batch_size, self.n_agents, length, -1).transpose(1, 2)
        if return_hidden:
            return outputs, hidden
        return outputs

    def build_step_inputs(self, obs, prev_actions_onehot):
        """Build policy inputs from obs [B,N,O] and previous proposals [B,N,A]."""
        inputs = [obs]
        if self.args.obs_last_action:
            if prev_actions_onehot is None:
                prev_actions_onehot = obs.new_zeros(*obs.shape[:-1], self.args.n_actions)
            inputs.append(prev_actions_onehot)
        if self.args.obs_agent_id:
            inputs.append(self._ids(obs).unsqueeze(0).expand(obs.shape[0], -1, -1))
        return th.cat(inputs, dim=-1).reshape(obs.shape[0] * self.n_agents, -1)

    def forward_step(self, obs, prev_actions_onehot, hidden_before):
        """Side-effect-free virtual rollout; returns Q and the advanced hidden."""
        outputs, hidden = self.agent(self.build_step_inputs(obs, prev_actions_onehot), hidden_before)
        return (outputs.reshape(obs.shape[0], self.n_agents, -1),
                hidden.reshape(obs.shape[0], self.n_agents, -1))

    def forward_q_attack(self, ep_batch, t, hidden_states, test_mode=False):
        outputs, _ = self.agent(self._build_forced_inputs(ep_batch, t), hidden_states)
        return outputs.reshape(ep_batch.batch_size, self.n_agents, -1)

    def forward_action(self, ep_batch, obs, pre_actions, hidden_state, attack=False, test_mode=False, forced=False):
        inputs = self._build_inputs_action(ep_batch, obs, pre_actions, attack, forced)
        outputs, _ = self.agent(inputs, hidden_state)
        if attack:
            return outputs.reshape(1, self.n_agents, -1)
        return outputs.reshape(ep_batch.batch_size, obs.shape[1], self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1).contiguous()
        self.last_hidden_before = None
        self.last_q_values = None

    def return_hidden(self):
        return self.hidden_states

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def to(self, device):
        self.agent.to(device)
        for name in ("hidden_states", "last_hidden_before", "last_q_values", "_agent_ids"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(device))
        return self

    def cuda(self):
        return self.to("cuda")

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location="cpu", weights_only=True))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _ids(self, reference):
        if self._agent_ids is None or self._agent_ids.device != reference.device or self._agent_ids.dtype != reference.dtype:
            self._agent_ids = th.eye(self.n_agents, device=reference.device, dtype=reference.dtype)
        return self._agent_ids

    def _build_inputs(self, batch, t):
        previous = None if t == 0 else batch["actions_onehot"][:, t - 1]
        return self.build_step_inputs(batch["obs"][:, t], previous)

    def _build_forced_inputs(self, batch, t):
        previous = None if t == 0 else batch["forced_actions_onehot"][:, t - 1]
        return self.build_step_inputs(batch["obs"][:, t], previous)

    def _build_inputs_action(self, batch, obs, pre_actions, attack=False, forced=False):
        leading = obs.shape[:-2]
        inputs = [obs]
        if self.args.obs_last_action:
            inputs.append(pre_actions)
        if self.args.obs_agent_id:
            ids = self._ids(obs).reshape(*([1] * len(leading)), self.n_agents, self.n_agents)
            inputs.append(ids.expand(*leading, self.n_agents, self.n_agents))
        return th.cat(inputs, dim=-1).reshape(-1, sum(x.shape[-1] for x in inputs))

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
