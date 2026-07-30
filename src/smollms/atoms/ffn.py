"""Feed-forward network (channel mixer) with SwiGLU.

Attention mixes *across tokens*. The FFN mixes *across features* of each token
independently — same MLP applied at every position.

Classic GPT FFN
---------------
    h = GELU(x W_up)
    y = h W_down

SwiGLU (Llama, Qwen, most modern open LLMs)
-------------------------------------------
    gate = SiLU(x W_gate)     # SiLU(z) = z * sigmoid(z)
    up   = x W_up
    h    = gate * up          # elementwise — the "GLU" gate
    y    = h W_down

So three matrices (gate, up, down) instead of two. The gate can zero out or
pass channels of the up-projection, which is more expressive than a plain GELU.

Width
-----
``hidden_dim`` (a.k.a. intermediate size) is usually ~ 8/3 * d_model, often
rounded to a multiple of 256 for hardware. For toys we just pass it explicitly.

Shape
-----
x, y: (B, T, d_model)  — FFN does not mix the T axis.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """Position-wise SwiGLU feed-forward network.

    Args:
        d_model: residual stream width.
        hidden_dim: expansion width (intermediate size).
        bias: whether linear layers use bias (Qwen/Llama: usually False).
        dropout: dropout on the hidden activation (0 for many modern recipes).
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")

        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) — usually already RMSNorm'd by the block.

        Returns:
            (B, T, d_model) update for residual_add.
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = self.drop(gate * up)
        return self.down_proj(hidden)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, hidden_dim={self.hidden_dim}"


def swiglu_hidden_dim(d_model: int, mult: float = 8 / 3, multiple_of: int = 8) -> int:
    """Common heuristic for intermediate size (toy-friendly multiple_of=8).

    Real models often use multiple_of=256. Formula mirrors Llama-style:
    round ``mult * d_model`` up to a multiple of ``multiple_of``.
    """
    hidden = int(mult * d_model)
    hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
    return hidden
