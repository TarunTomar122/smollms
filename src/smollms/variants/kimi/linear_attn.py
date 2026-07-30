"""Minimal Kimi Delta Attention (KDA) for the learning lab.

KDA keeps a fixed-size matrix state per head. At each token it decays old
key rows, reads the current key, then writes only the residual error:

    S <- decay * S
    error <- value - key^T S
    S <- S + beta * key outer error
    output <- query^T S

This is the delta-rule core missing from ordinary feature-map linear
attention. Production KDA adds convolutions, chunkwise kernels, and richer
parameterization; this direct recurrence makes the state update inspectable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def kda_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Run the causal KDA state update.

    All tensors are ``(batch, heads, time, head_dim)``. ``decay`` is
    per-key-channel in ``[0, 1]`` and ``beta`` controls each token's write.
    """
    b, h, t, d = q.shape
    state = q.new_zeros(b, h, d, d)
    outputs: list[torch.Tensor] = []

    # ponytail: direct recurrence teaches the update; use a chunkwise KDA kernel
    # only when sequence-length profiling, rather than this lab, needs it.
    for i in range(t):
        key = k[:, :, i]
        state = state * decay[:, :, i].unsqueeze(-1)
        predicted = torch.einsum("bhd,bhde->bhe", key, state)
        error = v[:, :, i] - predicted
        state = state + beta[:, :, i].unsqueeze(-1) * torch.einsum(
            "bhd,bhe->bhde", key, error
        )
        outputs.append(torch.einsum("bhd,bhde->bhe", q[:, :, i], state))

    return torch.stack(outputs, dim=2)


class KimiDeltaAttention(nn.Module):
    """Tiny KDA: projected Q/K/V, channel-wise decay, delta update, output gate."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.decay_proj = nn.Linear(d_model, d_model, bias=bias)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=bias)
        self.g_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape

        def heads(proj: nn.Linear) -> torch.Tensor:
            return proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        q = F.normalize(heads(self.q_proj), dim=-1, eps=1e-6)
        k = F.normalize(heads(self.k_proj), dim=-1, eps=1e-6)
        v = heads(self.v_proj)
        decay = torch.sigmoid(heads(self.decay_proj))
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2).unsqueeze(-1)

        y = kda_recurrence(q, k, v, decay, beta)
        y = y.transpose(1, 2).contiguous().view(b, t, self.d_model)
        y = self.drop(y) * torch.sigmoid(self.g_proj(x))
        return self.o_proj(y)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_heads={self.n_heads}"
