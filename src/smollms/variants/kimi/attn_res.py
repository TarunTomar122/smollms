"""Minimal Attention Residuals: content-dependent reads across earlier depths."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionResidual(nn.Module):
    """Let each token choose which earlier block stream to add back in.

    Attention is over depth, not sequence position: a token compares its
    current residual vector with that same token's vectors from earlier blocks.
    """

    def __init__(self, d_model: int, depth: int = 2, bias: bool = False) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.d_model = d_model
        self.depth = depth
        self.depth_bias = nn.Parameter(torch.zeros(depth))
        self.gate_bias = nn.Parameter(torch.tensor(-2.0))
        self.gate_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor, history: list[torch.Tensor]) -> torch.Tensor:
        if not history:
            return x

        recent = list(reversed(history[-self.depth :]))
        states = torch.stack(recent, dim=2)  # (B, T, earlier_depth, D)
        # ponytail: same-token depth attention; add cross-token reads only if a measured task needs them.
        query = F.normalize(x, dim=-1)
        keys = F.normalize(states, dim=-1)
        scores = torch.einsum("btd,btnd->btn", query, keys)
        scores = scores + self.depth_bias[: states.size(2)]
        weights = F.softmax(scores, dim=-1)
        context = torch.einsum("btn,btnd->btd", weights, states)
        gate = torch.sigmoid(self.gate_proj(x) + self.gate_bias)
        return x + gate * self.out_proj(context)
