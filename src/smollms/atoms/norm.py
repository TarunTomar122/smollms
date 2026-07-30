"""RMSNorm — the default pre-norm in modern LLMs (Llama, Qwen, ...).

What it does
------------
For each token vector x of size d:

    rms = sqrt(mean(x^2) + eps)
    y   = (x / rms) * weight

That is: scale by the root-mean-square of the features, then multiply by a
learned per-channel gain `weight` (often called gamma).

Why not LayerNorm?
------------------
LayerNorm also subtracts the mean:

    y = (x - mean(x)) / std(x) * weight + bias

RMSNorm drops the mean subtraction (and usually the bias). Empirically this is
enough for transformers, slightly cheaper, and is what almost every modern open
LLM uses.

Where it sits
-------------
Pre-norm residual blocks look like:

    x = x + attn(RMSNorm(x))
    x = x + ffn(RMSNorm(x))

Norm is applied *before* the sublayer, not after. That stabilizes deep stacks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Args:
        dim: feature size (last dimension of the residual stream).
        eps: small constant so we never divide by zero.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # Learned scale, one value per feature channel. Starts at 1 = "do nothing".
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)  — typically (batch, seq, dim)
        # Compute in float32 for numerical stability even if x is float16/bfloat16.
        x_f = x.float()
        # mean over the last dim only — each token is normalized independently.
        variance = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_f * torch.rsqrt(variance + self.eps)
        # Cast back to original dtype, then apply learned gain.
        return self.weight * x_normed.type_as(x)

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}, eps={self.eps}"
