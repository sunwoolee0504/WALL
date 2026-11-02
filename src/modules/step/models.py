"""Causal, padding-aware WALL predictors using a new predictor checkpoint format."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p, backend="sdpa"):
        super().__init__()
        if h_dim % n_heads:
            raise ValueError("trans_embed_dim must be divisible by num_trans_head")
        if backend not in ("sdpa", "math", "auto"):
            raise ValueError("attention_backend must be sdpa, auto, or math")
        self.n_heads = n_heads
        self.drop_p = drop_p
        self.backend = backend
        self.qkv = nn.Linear(h_dim, 3 * h_dim)
        self.proj_net = nn.Linear(h_dim, h_dim)
        self.proj_drop = nn.Dropout(drop_p)

    def forward(self, x, valid_mask=None):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(
            batch, length, 3, self.n_heads, width // self.n_heads
        ).permute(2, 0, 3, 1, 4).unbind(0)
        allowed = None
        if valid_mask is not None or self.backend == "math":
            causal = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
            allowed = causal[None, None]
            if valid_mask is not None:
                valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)
                allowed = allowed & valid_mask[:, None, None, :]
                # Invalid queries attend only to themselves, avoiding all-inf
                # softmax rows; their outputs are explicitly zeroed below.
                diagonal = torch.eye(length, device=x.device, dtype=torch.bool)
                allowed = (allowed & valid_mask[:, None, :, None]) | (
                    diagonal[None, None] & ~valid_mask[:, None, :, None]
                )
        dropout = self.drop_p if self.training else 0.0
        if self.backend == "math":
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            scores = scores.masked_fill(~allowed, -torch.inf)
            weights = F.dropout(scores.softmax(dim=-1), p=dropout, training=self.training)
            attention = weights @ v
        else:
            attention = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed, dropout_p=dropout,
                is_causal=allowed is None,
            )
        attention = attention.transpose(1, 2).reshape(batch, length, width)
        result = self.proj_drop(self.proj_net(attention))
        if valid_mask is not None:
            result = result.masked_fill(~valid_mask[..., None], 0)
        return result


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p, ff_multiplier=4, backend="sdpa"):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p, backend)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, int(ff_multiplier * h_dim)), nn.GELU(),
            nn.Linear(int(ff_multiplier * h_dim), h_dim), nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x, valid_mask=None):
        x = x + self.attention(self.ln1(x), valid_mask)
        x = x + self.mlp(self.ln2(x))
        if valid_mask is not None:
            x = x.masked_fill(~valid_mask[..., None], 0)
        return x


class CausalBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.width = args.trans_embed_dim
        self.register_buffer(
            "inverse_frequency",
            torch.exp(torch.arange(0, self.width, 2).float() * (-math.log(10000.0) / self.width)),
            persistent=False,
        )
        dropout = getattr(args, "trans_dropout", 0.1)
        self.input_norm = nn.LayerNorm(self.width)
        self.blocks = nn.ModuleList([
            Block(self.width, args.trans_input_len, args.num_trans_head, dropout,
                  getattr(args, "trans_ff_multiplier", 4), getattr(args, "attention_backend", "sdpa"))
            for _ in range(args.num_trans_layer)
        ])
        self.output_norm = nn.LayerNorm(self.width)

    def forward(self, embeddings, timesteps=None, valid_mask=None, last_only=False):
        batch, length, _ = embeddings.shape
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=embeddings.device, dtype=torch.bool)
        if timesteps is None:
            if valid_mask is None:
                timesteps = torch.arange(length, device=embeddings.device)[None].expand(batch, -1)
            else:
                timesteps = (valid_mask.long().cumsum(dim=-1) - 1).clamp_min(0)
        angles = timesteps.to(device=embeddings.device, dtype=torch.float32)[..., None] * self.inverse_frequency
        positions = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)[..., :self.width]
        x = self.input_norm(embeddings + positions.to(embeddings.dtype))
        if valid_mask is not None:
            x = x.masked_fill(~valid_mask[..., None], 0)
        for block in self.blocks:
            x = block(x, valid_mask)
        if last_only:
            x = x[:, -1:]
            if valid_mask is not None:
                valid_mask = valid_mask[:, -1:]
        x = self.output_norm(x)
        if valid_mask is not None:
            x = x.masked_fill(~valid_mask[..., None], 0)
        return x


class Qdifference_Transformer(nn.Module):
    """Expected attack effect over uniformly sampled initial agents.

    Inputs are absolute timestamps [B,T] and state history [B,T,S]. A single
    backbone predicts all attack-time offsets [B,T,attack_period].
    """
    def __init__(self, args):
        super().__init__()
        self.embed_state = nn.Linear(args.state_shape, args.trans_embed_dim)
        self.backbone = CausalBackbone(args)
        self.predict_qdiff = nn.Linear(args.trans_embed_dim, args.attack_period)

    def forward(self, timesteps, state, valid_mask=None, last_only=False):
        hidden = self.backbone(self.embed_state(state), timesteps, valid_mask, last_only=last_only)
        if last_only and valid_mask is not None:
            valid_mask = valid_mask[:, -1:]
        output = self.predict_qdiff(hidden)
        if valid_mask is not None:
            output = output.masked_fill(~valid_mask.to(device=state.device, dtype=torch.bool)[..., None], 0)
        return output


class Planning_Transformer(nn.Module):
    """Predict next state/observations from past state, observations and actions."""
    def __init__(self, args):
        super().__init__()
        self.n_agents = args.n_agents
        self.obs_dim = args.obs_shape
        self.n_actions = args.n_actions
        width = args.trans_embed_dim
        self.state_embed = nn.Linear(args.state_shape, width)
        self.obs_embed = nn.Linear(args.n_agents * args.obs_shape, width)
        self.action_embed = nn.Linear(args.n_agents * args.n_actions, width)
        self.backbone = CausalBackbone(args)
        self.next_state = nn.Linear(width, args.state_shape)
        # Joint heads give each agent a context-dependent observation output.
        self.next_obs = nn.Linear(width, args.n_agents * args.obs_shape)
        self.next_availability = nn.Linear(width, args.n_agents * args.n_actions)
        self.next_termination = nn.Linear(width, 1)

    def forward(self, state, obs, actions_onehot, timesteps=None, valid_mask=None, return_aux=False, last_only=False):
        batch, length = state.shape[:2]
        embeddings = (
            self.state_embed(state)
            + self.obs_embed(obs.reshape(batch, length, -1))
            + self.action_embed(actions_onehot.reshape(batch, length, -1))
        ) / math.sqrt(3.0)
        hidden = self.backbone(embeddings, timesteps, valid_mask, last_only=last_only)
        if last_only:
            length = 1
            if valid_mask is not None:
                valid_mask = valid_mask[:, -1:]
        next_state = self.next_state(hidden)
        next_obs = self.next_obs(hidden).view(batch, length, self.n_agents, self.obs_dim)
        outputs = (next_state, next_obs)
        if return_aux:
            availability = self.next_availability(hidden).view(batch, length, self.n_agents, self.n_actions)
            outputs += (availability, self.next_termination(hidden))
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=state.device, dtype=torch.bool)
            outputs = tuple(x.masked_fill(~valid_mask.reshape(batch, length, *([1] * (x.ndim - 2))), 0) for x in outputs)
        return outputs
