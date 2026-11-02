"""Model semantics that must hold independently of StarCraft installation."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from modules.agents.rnn_agent import RNNAgent
from modules.step.models import Planning_Transformer, Qdifference_Transformer
from controllers.basic_controller import BasicMAC
from utils.sequence import gather_history


def args(**overrides):
    values = dict(state_shape=7, obs_shape=5, n_agents=3, n_actions=4,
                  trans_embed_dim=16, num_trans_layer=2, trans_input_len=6,
                  num_trans_head=4, attack_period=4, trans_dropout=0.0,
                  trans_ff_multiplier=4, attention_backend="sdpa", rnn_hidden_dim=12,
                  obs_last_action=True, obs_agent_id=True, agent="rnn",
                  agent_output_type="q", action_selector="epsilon_greedy",
                  epsilon_start=1.0, epsilon_finish=0.05, epsilon_anneal_time=5000)
    values.update(overrides)
    return SimpleNamespace(**values)


def sample_inputs(batch=2, length=6):
    return (torch.randn(batch, length, 7), torch.randn(batch, length, 3, 5),
            torch.randn(batch, length, 3, 4),
            torch.arange(1200, 1200 + length).expand(batch, -1))


@pytest.mark.parametrize("backend", ["sdpa", "math"])
def test_predictors_cannot_see_future(backend):
    torch.manual_seed(4)
    planner = Planning_Transformer(args(attention_backend=backend)).eval()
    qdiff = Qdifference_Transformer(args(attention_backend=backend)).eval()
    state, obs, action, times = sample_inputs()
    original = planner(state, obs, action, times, return_aux=True)
    q_original = qdiff(times, state)
    changed = [x.clone() for x in (state, obs, action)]
    for value in changed:
        value[:, 3:] = torch.randn_like(value[:, 3:]) * 20
    perturbed = planner(*changed, times, return_aux=True)
    for before, after in zip(original, perturbed):
        torch.testing.assert_close(before[:, :3], after[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(q_original[:, :3], qdiff(times, changed[0])[:, :3], rtol=0, atol=0)
    state.requires_grad_()
    planner(state, obs, action, times)[0][:, 2].sum().backward()
    assert torch.count_nonzero(state.grad[:, 3:]) == 0


@pytest.mark.parametrize("backend", ["sdpa", "math"])
def test_padding_is_invisible_and_all_pad_gradients_finite(backend):
    torch.manual_seed(5)
    planner = Planning_Transformer(args(attention_backend=backend)).eval()
    qdiff = Qdifference_Transformer(args(attention_backend=backend)).eval()
    state, obs, action, times = sample_inputs(batch=1, length=3)
    padded = [torch.cat((torch.randn_like(x) * 100, x), dim=1) for x in (state, obs, action)]
    padded_times = torch.cat((torch.zeros_like(times), times), dim=1)
    valid = torch.tensor([[False, False, False, True, True, True]])
    reference = planner(state, obs, action, times, return_aux=True)
    result = planner(*padded, padded_times, valid, return_aux=True)
    for before, after in zip(reference, result):
        torch.testing.assert_close(before, after[:, 3:], rtol=1e-5, atol=1e-6)
        assert torch.count_nonzero(after[:, :3]) == 0
    torch.testing.assert_close(qdiff(times, state), qdiff(padded_times, padded[0], valid)[:, 3:], rtol=1e-5, atol=1e-6)
    # Also exercise mixed empty/nonempty rows in the same training minibatch.
    values = [torch.cat((x, x), dim=0).requires_grad_() for x in padded]
    mixed_valid = torch.cat((torch.zeros_like(valid), valid), dim=0)
    outputs = planner(*values, padded_times.expand(2, -1), mixed_valid, return_aux=True)
    loss = sum(output.square().sum() for output in outputs)
    loss.backward()
    assert all(torch.isfinite(output).all() for output in outputs)
    assert all(torch.isfinite(value.grad).all() for value in values)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in planner.parameters())


def test_fused_attention_matches_reference_backend():
    torch.manual_seed(6)
    fused = Planning_Transformer(args()).eval()
    reference = Planning_Transformer(args(attention_backend="math")).eval()
    reference.load_state_dict(fused.state_dict())
    state, obs, action, times = sample_inputs()
    valid = torch.tensor([[False, False, True, True, True, True], [False] * 6])
    for mask in (None, valid):
        a = fused(state, obs, action, times, mask, return_aux=True)
        b = reference(state, obs, action, times, mask, return_aux=True)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)


def test_agent_observation_differences_depend_on_context():
    torch.manual_seed(7)
    planner = Planning_Transformer(args()).eval()
    state, obs, action, times = sample_inputs()
    predictions = planner(state, obs, action, times)[1]
    differences = predictions[:, :, 0] - predictions[:, :, 1]
    assert not torch.allclose(differences[:, 0], differences[:, -1], atol=1e-5)
    assert planner(state, obs, action, times, return_aux=True)[2].shape == (2, 6, 3, 4)


class LegacyAgent(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, config.rnn_hidden_dim)
        self.rnn = nn.GRUCell(config.rnn_hidden_dim, config.rnn_hidden_dim)
        self.fc2 = nn.Linear(config.rnn_hidden_dim, config.n_actions)

    def forward(self, inputs, hidden):
        hidden = self.rnn(self.fc1(inputs).relu(), hidden)
        return self.fc2(hidden), hidden


def test_legacy_grucell_checkpoint_sequence_outputs_and_gradients_match():
    torch.manual_seed(9)
    config = args()
    legacy = LegacyAgent(11, config)
    agent = RNNAgent(11, config)
    agent.load_state_dict(legacy.state_dict(), strict=True)
    inputs = torch.randn(6, 7, 11)
    h = torch.randn(6, 12)
    old_outputs, old_hidden = [], []
    current = h
    for t in range(inputs.size(1)):
        q, current = legacy(inputs[:, t], current)
        old_outputs.append(q)
        old_hidden.append(current)
    old_outputs = torch.stack(old_outputs, dim=1)
    old_hidden = torch.stack(old_hidden, dim=1)
    q_sequence, hidden_sequence = agent.forward_sequence(inputs, h)
    torch.testing.assert_close(q_sequence, old_outputs, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(hidden_sequence, old_hidden, rtol=1e-5, atol=1e-6)
    q_step, h_step = agent(inputs[:, 0], h)
    torch.testing.assert_close(q_step, old_outputs[:, 0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(h_step, old_hidden[:, 0], rtol=1e-5, atol=1e-6)
    q_sequence.square().sum().backward()
    old_outputs.square().sum().backward()
    gradients = dict(agent.named_parameters())
    for name, parameter in legacy.named_parameters():
        new_name = name + "_l0" if name.startswith("rnn.") else name
        torch.testing.assert_close(parameter.grad, gradients[new_name].grad, rtol=1e-4, atol=1e-5)


class Batch(dict):
    @property
    def batch_size(self):
        return self["obs"].size(0)


@pytest.mark.parametrize("last_action,agent_id", [(False, False), (False, True), (True, False), (True, True)])
def test_controller_sequence_matches_online_and_virtual_steps(last_action, agent_id, tmp_path):
    torch.manual_seed(10)
    config = args(obs_last_action=last_action, obs_agent_id=agent_id)
    scheme = {"obs": {"vshape": 5}, "actions_onehot": {"vshape": (4,)}}
    mac = BasicMAC(scheme, None, config)
    batch = Batch(obs=torch.randn(2, 6, 3, 5), actions_onehot=torch.randn(2, 6, 3, 4),
                  avail_actions=torch.ones(2, 6, 3, 4))
    mac.init_hidden(2)
    initial = mac.return_hidden().clone()
    sequence_q, sequence_h = mac.forward_sequence(batch, return_hidden=True)
    assert torch.equal(mac.return_hidden(), initial)
    for t in range(6):
        before = mac.return_hidden()
        actual = mac.forward(batch, t)
        prev_action = None if t == 0 else batch["actions_onehot"][:, t - 1]
        virtual_q, virtual_h = mac.forward_step(batch["obs"][:, t], prev_action, before)
        torch.testing.assert_close(sequence_q[:, t], actual, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(sequence_h[:, t], mac.return_hidden(), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(virtual_q, actual)
        torch.testing.assert_close(virtual_h, mac.return_hidden())
    mac.save_models(tmp_path)
    restored = BasicMAC(scheme, None, config)
    restored.load_models(tmp_path)
    torch.testing.assert_close(restored.forward_sequence(batch), sequence_q)


def test_gather_histories_matches_absolute_time_and_source_validity():
    source = torch.arange(2 * 6 * 2).float().reshape(2, 6, 2).requires_grad_()
    endpoints = torch.tensor([[0, 4], [2, 7]])
    valid_source = torch.tensor([[True] * 6, [True, True, True, True, False, False]])
    windows, valid, times = gather_history(source, endpoints, 4, valid_source)
    assert windows.shape == (2, 2, 4, 2)
    assert times[0, 1].tolist() == [1, 2, 3, 4]
    assert valid[0, 0].tolist() == [False, False, False, True]
    assert not valid[1, 1].any()
    assert torch.count_nonzero(windows[1, 1]) == 0
    torch.testing.assert_close(windows[0, 1], source[0, 1:5])
    windows.sum().backward()
    assert torch.isfinite(source.grad).all()
    assert source.grad[1, 4:].count_nonzero() == 0


@pytest.mark.parametrize("kind", ["planner", "qdiff"])
@pytest.mark.parametrize("backend", ["sdpa", "math"])
@pytest.mark.parametrize("training", [False, True])
def test_last_only_matches_full_outputs_and_gradients(kind, backend, training):
    import copy
    torch.manual_seed(91)
    settings = args(attention_backend=backend, trans_dropout=0.1 if training else 0.0)
    model = (Planning_Transformer(settings) if kind == "planner" else Qdifference_Transformer(settings)).train(training)
    fast = copy.deepcopy(model)
    state, obs, action, times = sample_inputs()
    valid = torch.tensor([[False, False, True, True, True, True], [True, True, True, False, False, False]])
    def outputs(module, last):
        torch.manual_seed(713)  # Dropout draws must remain unchanged.
        if kind == "planner":
            return module(state, obs, action, times, valid, return_aux=True, last_only=last)
        return (module(times, state, valid, last_only=last),)
    full = tuple(x[:, -1:] for x in outputs(model, False))
    reduced = outputs(fast, True)
    for a, b in zip(full, reduced):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        assert b[1].count_nonzero() == 0
    sum(x.square().sum() for x in full).backward()
    sum(x.square().sum() for x in reduced).backward()
    for a, b in zip(model.parameters(), fast.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=3e-5, atol=3e-6)
