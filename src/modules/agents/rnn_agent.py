import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    """One GRU parameter set shared by batched training and online execution."""

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRU(args.rnn_hidden_dim, args.rnn_hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new_zeros(1, self.args.rnn_hidden_dim)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        hidden = hidden_state.reshape(1, -1, self.args.rnn_hidden_dim).contiguous()
        output, hidden_after = self.rnn(x.unsqueeze(1), hidden)
        return self.fc2(output[:, 0]), hidden_after[0]

    def forward_sequence(self, inputs, hidden_state=None):
        """inputs [batch*agents,T,input]; return Q and hidden after every step."""
        x = F.relu(self.fc1(inputs))
        hidden = None if hidden_state is None else hidden_state.reshape(1, inputs.size(0), -1).contiguous()
        outputs, _ = self.rnn(x, hidden)
        return self.fc2(outputs), outputs

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # nn.GRU and GRUCell have identical r/z/n gate ordering/tensor layouts.
        # Translate legacy defender checkpoints before recursive module loading.
        for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            old = prefix + "rnn." + name
            new = old + "_l0"
            if old in state_dict:
                if new not in state_dict:
                    state_dict[new] = state_dict[old]
                del state_dict[old]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
