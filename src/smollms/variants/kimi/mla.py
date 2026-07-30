"""Minimal Gated Multi-Head Latent Attention for the Kimi learning variant."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from smollms.atoms.attention import causal_mask
from smollms.atoms.norm import RMSNorm
from smollms.atoms.rope import RotaryEmbedding, apply_rope


class GatedMultiHeadLatentAttention(nn.Module):
    """Causal attention whose K/V information passes through one token latent.

    This keeps the useful MLA idea small: store a compact latent per token,
    then reconstruct the per-head K/V vectors used by attention.  The output
    gate is deliberately explicit so its effect is easy to inspect.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        qk_norm: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.latent_dim = d_model // 2

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.kv_down = nn.Linear(d_model, self.latent_dim, bias=bias)
        self.k_up = nn.Linear(self.latent_dim, d_model, bias=bias)
        self.v_up = nn.Linear(self.latent_dim, d_model, bias=bias)
        self.g_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, base=rope_base)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        latent = self.kv_down(x)
        # ponytail: reconstruct K/V for the lesson; cache `latent` only when a decode path exists.
        k = self.k_up(latent).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(latent).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        cos, sin = self.rope(t)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + causal_mask(t, device=x.device, dtype=scores.dtype)
        weights = self.attn_drop(F.softmax(scores, dim=-1))
        y = torch.matmul(weights, v).transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.o_proj(y * torch.sigmoid(self.g_proj(x)))
