"""Minimal DeepSeek Sparse Attention: learned causal top-k token selection."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from smollms.atoms.attention import causal_mask
from smollms.atoms.norm import RMSNorm
from smollms.atoms.rope import RotaryEmbedding, apply_rope


class DeepSeekSparseAttention(nn.Module):
    """Use a cheap indexer to choose causal tokens for normal attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        top_k: int,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        qk_norm: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.index_dim = max(1, self.head_dim // 2)
        self.top_k = top_k

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.index_q_proj = nn.Linear(d_model, n_heads * self.index_dim, bias=bias)
        self.index_k_proj = nn.Linear(d_model, n_heads * self.index_dim, bias=bias)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, base=rope_base)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.attn_drop = nn.Dropout(dropout)
        self.last_selected_indices: torch.Tensor | None = None
        self.last_selected_valid: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        cos, sin = self.rope(t)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        index_q = self.index_q_proj(x).view(b, t, self.n_heads, self.index_dim).transpose(1, 2)
        index_k = self.index_k_proj(x).view(b, t, self.n_heads, self.index_dim).transpose(1, 2)
        index_scores = torch.matmul(index_q, index_k.transpose(-2, -1))
        index_scores = index_scores / math.sqrt(self.index_dim)
        index_scores = index_scores + causal_mask(t, device=x.device, dtype=index_scores.dtype)

        selected_indices = torch.topk(index_scores, k=min(self.top_k, t), dim=-1).indices
        positions = torch.arange(t, device=x.device).view(1, 1, t, 1)
        selected_valid = selected_indices <= positions
        self.last_selected_indices = selected_indices.detach()
        self.last_selected_valid = selected_valid.detach()

        batch = torch.arange(b, device=x.device)[:, None, None, None]
        heads = torch.arange(self.n_heads, device=x.device)[None, :, None, None]
        # ponytail: this materializes dense index scores for the lesson; use tiled kernels and a KV cache only for long contexts.
        selected_k = k[batch, heads, selected_indices]
        selected_v = v[batch, heads, selected_indices]
        selected_index_scores = torch.gather(index_scores, -1, selected_indices)

        scores = torch.einsum("bhtd,bhtkd->bhtk", q, selected_k)
        scores = scores / math.sqrt(self.head_dim) + selected_index_scores
        scores = scores.masked_fill(~selected_valid, torch.finfo(scores.dtype).min)
        weights = self.attn_drop(F.softmax(scores, dim=-1))
        y = torch.einsum("bhtk,bhtkd->bhtd", weights, selected_v)
        y = y.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.o_proj(y)
