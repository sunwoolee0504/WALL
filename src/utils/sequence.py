"""Gather only selected causal histories, with common timestamp/padding rules."""
import torch


def gather_history(tensor, endpoints, length, valid_mask=None):
    """Return ``(windows, valid, timesteps)`` for histories ending at endpoints.

    ``tensor`` is [B,T,...], ``endpoints`` is [B,K], and outputs are [B,K,L,...],
    [B,K,L], [B,K,L]. Endpoints are inclusive. Out-of-range or invalid source
    entries are zero-filled; padded timestamps are zero. ``valid_mask`` can be
    [B,T] or [B,T,1]. No [B,T,L,...] intermediate is constructed.
    """
    if tensor.ndim < 2 or endpoints.ndim != 2 or endpoints.shape[0] != tensor.shape[0]:
        raise ValueError("Expected tensor[B,T,...] and endpoints[B,K]")
    if length <= 0 or tensor.shape[1] == 0:
        raise ValueError("History length and source time dimension must be positive")
    endpoints = endpoints.to(device=tensor.device, dtype=torch.long)
    times = endpoints[..., None] + torch.arange(1 - length, 1, device=tensor.device)
    valid = (times >= 0) & (times < tensor.shape[1])
    indices = times.clamp(0, tensor.shape[1] - 1)
    batches = torch.arange(tensor.shape[0], device=tensor.device)[:, None, None]
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=tensor.device, dtype=torch.bool).reshape(tensor.shape[:2])
        valid = valid & valid_mask[batches, indices]
    windows = tensor[batches, indices]
    windows = windows.masked_fill(~valid.reshape(*valid.shape, *([1] * (tensor.ndim - 2))), 0)
    return windows, valid, times.masked_fill(~valid, 0)
