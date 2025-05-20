import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        self.register_buffer('mask',mask)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x) # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x


class Qdifference_Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.state_dim = args.state_shape
        self.trans_embed_dim = args.trans_embed_dim
        self.num_trans_layer = args.num_trans_layer
        self.trans_input_len = args.trans_input_len
        self.num_trans_head = args.num_trans_head
        self.drop_p = 0.1  # Dropout probability

        # Transformer blocks
        self.transformer = nn.Sequential(
            *[Block(self.trans_embed_dim, self.trans_input_len, self.num_trans_head, self.drop_p)
              for _ in range(self.num_trans_layer)]
        )

        # Embedding layers
        self.embed_ln = nn.LayerNorm(self.trans_embed_dim)
        self.embed_state = nn.Linear(self.state_dim, self.trans_embed_dim)
        self.embed_timestep = nn.Embedding(args.env_max_timestep, self.trans_embed_dim)

        # Q-difference prediction layer
        self.predict_qdiff = nn.Linear(self.trans_embed_dim + args.n_agents, args.attack_period)

    def forward(self, timesteps, state, target_agent_id):
        time_embeddings = self.embed_timestep(timesteps.long())
        state_embeddings = self.embed_state(state) + time_embeddings  # Shape: (batch, seq_len, embed_dim)
        h = self.embed_ln(state_embeddings)
        h = self.transformer(h)

        x = torch.cat((h, target_agent_id), dim=-1)
        return self.predict_qdiff(x)


class Planning_Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.num_trans_head = args.num_trans_head
        self.num_trans_layer = args.num_trans_layer
        self.trans_embed_dim = args.trans_embed_dim

        # Input dimensions
        self.state_dim = args.state_shape
        self.obs_dim = args.obs_shape
        self.action_dim = args.n_actions * args.n_agents

        # Embedding layers
        self.state_embed = nn.Linear(self.state_dim, self.trans_embed_dim)
        self.history_embed = nn.Linear(self.trans_embed_dim * args.n_agents, self.trans_embed_dim)
        self.action_embed = nn.Linear(self.action_dim, self.trans_embed_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.trans_embed_dim * 3, nhead=self.num_trans_head
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_trans_layer)

        # Output layers
        self.fc_next_state = nn.Linear(self.trans_embed_dim * 3, self.state_dim)
        self.fc_next_obc_embed = nn.Linear(self.trans_embed_dim * 3, self.trans_embed_dim)
        self.fc_obs = nn.Linear(self.trans_embed_dim + args.n_agents, self.obs_dim)

        self.relu = nn.ReLU()

    def forward(self, state, history, action):
        batch_size, seq_len = state.shape[:2]

        joint_action = action.view(batch_size, seq_len, -1)
        joint_history = history.view(batch_size, seq_len, -1)

        state_emb = self.relu(self.state_embed(state))
        history_emb = self.relu(self.history_embed(joint_history))
        action_emb = self.relu(self.action_embed(joint_action))

        tgt = torch.cat((state_emb, history_emb, action_emb), dim=-1).permute(1, 0, 2)
        transformer_out = self.transformer_decoder(tgt, memory=tgt)

        next_state = self.fc_next_state(transformer_out).permute(1, 0, 2)
        next_embed = self.fc_next_obc_embed(transformer_out).permute(1, 0, 2)

        return next_state, next_embed

    def get_next_obs(self, obs_embed, target_agent_id):
        x = torch.cat((obs_embed, target_agent_id), dim=-1)
        return self.fc_obs(self.relu(x))
