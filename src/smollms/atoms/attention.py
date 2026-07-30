"""Multi-head causal attention (MHA / GQA) with RoPE — dense Qwen-style core.

Attention in one sentence
-------------------------
Each token builds a **query**, every token offers **keys** and **values**;
the query takes a softmax-weighted sum of values (only past + self, for LMs).

    scores_ij = (q_i · k_j) / sqrt(head_dim)
    weights   = softmax(scores_i, over j ≤ i)     # causal
    out_i     = sum_j  weights_ij * v_j

Multi-head: run this in ``n_heads`` parallel subspaces, then concat + project.

GQA (Grouped-Query Attention)
-----------------------------
Full MHA: one K/V head per Q head.
GQA: fewer K/V heads; several Q heads share the same K/V head.
Qwen / Llama use this to shrink the KV cache without killing quality.

QK-Norm (Qwen3)
---------------
Optional RMSNorm on q and k *per head* before RoPE — stabilizes training.
We expose ``qk_norm=True`` so our dense-Qwen path can match that recipe.

Shapes (typical)
----------------
x:     (B, T, d_model)
q:     (B, n_heads,    T, head_dim)
k,v:   (B, n_kv_heads, T, head_dim)  then expanded for GQA
out:   (B, T, d_model)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from smollms.atoms.norm import RMSNorm
from smollms.atoms.rope import RotaryEmbedding, apply_rope


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA: (B, n_kv, T, D) → (B, n_kv * n_rep, T, D)."""
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    # insert axis, repeat, merge back into head axis
    x = x[:, :, None, :, :].expand(b, n_kv, n_rep, t, d)
    return x.reshape(b, n_kv * n_rep, t, d)


def causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """(1, 1, T, T) mask with 0 on/below diagonal and -inf above (future)."""
    # True where we *block* (j > i)
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )
    # additive mask for scores
    out = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    out = out.masked_fill(mask, torch.finfo(dtype).min)
    return out[None, None, :, :]


class CausalSelfAttention(nn.Module):
    """Causal multi-head attention with RoPE and optional GQA + QK-Norm.

    Args:
        d_model: residual stream width.
        n_heads: number of query heads.
        n_kv_heads: number of key/value heads (GQA). Defaults to ``n_heads`` (MHA).
        max_seq_len: RoPE cache length.
        rope_base: RoPE θ base (10000 is the classic default).
        qk_norm: if True, RMSNorm each head's q and k (Qwen3-style).
        dropout: attention dropout (0 for toy / eval-style training often).
        bias: whether QKV / output projections use bias (Qwen: usually False).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        qk_norm: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if n_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.qk_norm_enabled = qk_norm

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=bias)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, base=rope_base)

        if qk_norm:
            # Normalize within each head (last dim = head_dim)
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        self.dropout = dropout
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) — usually already RMSNorm'd by the block.

        Returns:
            (B, T, d_model) attention output (delta to residual-add).
        """
        b, t, _ = x.shape

        q = self.q_proj(x)  # (B, T, n_heads * head_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # q: (B, n_heads, T, head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos, sin = self.rope(t)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        # k,v: (B, n_heads, T, head_dim)

        # scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        # (B, h, T, d) @ (B, h, d, T) → (B, h, T, T)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn + causal_mask(t, device=x.device, dtype=attn.dtype)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        y = torch.matmul(attn, v)  # (B, h, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.o_proj(y)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_kv_heads={self.n_kv_heads}, head_dim={self.head_dim}, "
            f"qk_norm={self.qk_norm_enabled}"
        )
