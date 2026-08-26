"""Mask-aware spectrum embedding.

DeepSets-style: per-channel MLP over (wavelength, depth, error) tokens,
masked mean+max pooling, then a head MLP. Permutation-invariant over
channels and indifferent to grid length — the property that lets one
network serve different binnings, reduction variants and masked
channels. Upgrade path to attention pooling / transformer is isolated
here; nothing downstream changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# normalization constants for the G395H sub-Neptune regime
_LAM_LO, _LAM_HI = 2.7, 5.3
_DEPTH_SCALE = 100.0     # depth ~ 1e-3..7e-3 -> 0.1..0.7
_ERR_SCALE = 1.0e4       # err ~ 3e-5..3e-4 -> 0.3..3


def _normalize(tokens: torch.Tensor) -> torch.Tensor:
    lam = (tokens[..., 0] - _LAM_LO) / (_LAM_HI - _LAM_LO)
    depth = tokens[..., 1] * _DEPTH_SCALE
    err = tokens[..., 2] * _ERR_SCALE
    return torch.stack([lam, depth, err], dim=-1)


class SpectrumEmbedding(nn.Module):
    def __init__(self, out_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.channel_mlp = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """tokens (b, m, 3), mask (b, m) bool -> (b, out_dim)."""
        x = self.channel_mlp(_normalize(tokens))          # (b, m, h)
        w = mask.unsqueeze(-1).to(x.dtype)
        mean = (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        neg = torch.finfo(x.dtype).min
        maxed = x.masked_fill(~mask.unsqueeze(-1), neg).amax(dim=1)
        return self.head(torch.cat([mean, maxed], dim=-1))
