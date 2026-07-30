"""Causal local attention plus compressed older memory: V4 teaching step 1."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from smollms.atoms.norm import RMSNorm
from smollms.atoms.rope import RotaryEmbedding, apply_rope


class LocalCompressedAttention(nn.Module):
    """Read a raw local window and learned summaries of the older prefix.

    This is deliberately smaller than DeepSeek V4: it has no sparse indexer.
    The boundary raw read ensures compression never drops an old token merely
    because the sliding window moved through a chunk.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        local_window: int = 16,
        compression_ratio: int = 4,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        qk_norm: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if local_window < 1 or compression_ratio < 1:
            raise ValueError("local_window and compression_ratio must be >= 1")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.local_window = local_window
        self.compression_ratio = compression_ratio
        self.max_seq_len = max_seq_len
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, base=rope_base)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.compress_k = nn.Linear(compression_ratio * self.head_dim, self.head_dim, bias=bias)
        self.compress_v = nn.Linear(compression_ratio * self.head_dim, self.head_dim, bias=bias)
        self.attn_drop = nn.Dropout(dropout)

    def memory_masks(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return valid compressed chunks and raw boundary tokens per query."""
        pos = torch.arange(seq_len, device=device)
        local_start = (pos - self.local_window + 1).clamp_min(0)
        n_chunks = seq_len // self.compression_ratio
        chunk_end = (
            torch.arange(n_chunks, device=device) * self.compression_ratio
            + self.compression_ratio
            - 1
        )
        compressed = chunk_end[None, :] < local_start[:, None]
        full_prefix_end = (local_start // self.compression_ratio) * self.compression_ratio
        boundary = (pos[None, :] < local_start[:, None]) & (
            pos[None, :] >= full_prefix_end[:, None]
        )
        return compressed, boundary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        if t > self.max_seq_len:
            raise ValueError(f"sequence length {t} > max_seq_len {self.max_seq_len}")
        h, d = self.n_heads, self.head_dim
        q = self.q_proj(x).view(b, t, h, d).transpose(1, 2)
        k = self.k_proj(x).view(b, t, h, d).transpose(1, 2)
        v = self.v_proj(x).view(b, t, h, d).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = self.rope(t)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        pos = torch.arange(t, device=x.device)
        local_valid = (pos[None, :] <= pos[:, None]) & (
            pos[None, :] >= (pos[:, None] - self.local_window + 1)
        )
        scale = 1.0 / math.sqrt(d)
        local_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        local_scores = local_scores.masked_fill(~local_valid[None, None], torch.finfo(q.dtype).min)
        local_out = torch.matmul(self.attn_drop(F.softmax(local_scores, dim=-1)), v)

        n_chunks = t // self.compression_ratio
        if n_chunks:
            chunk_width = self.compression_ratio * d
            mem_k = self.compress_k(k[:, :, : n_chunks * self.compression_ratio].reshape(b, h, n_chunks, chunk_width))
            mem_v = self.compress_v(v[:, :, : n_chunks * self.compression_ratio].reshape(b, h, n_chunks, chunk_width))
        else:
            mem_k, mem_v = k[:, :, :0], v[:, :, :0]
        # ponytail: dense masks make this O(T^2) teaching code; use chunked kernels
        # and a KV cache before treating local/compressed attention as a speed feature.
        memory_k, memory_v = torch.cat((mem_k, k), dim=2), torch.cat((mem_v, v), dim=2)
        compressed_valid, boundary_valid = self.memory_masks(t, x.device)
        memory_valid = torch.cat((compressed_valid, boundary_valid), dim=-1)
        memory_scores = torch.matmul(q, memory_k.transpose(-2, -1)) * scale
        memory_scores = memory_scores.masked_fill(
            ~memory_valid[None, None], torch.finfo(q.dtype).min
        )
        memory_weights = F.softmax(memory_scores, dim=-1)
        has_memory = memory_valid.any(dim=-1, keepdim=True)[None, None]
        memory_weights = torch.where(has_memory, memory_weights, torch.zeros_like(memory_weights))
        memory_out = torch.matmul(self.attn_drop(memory_weights), memory_v)

        out = (local_out + memory_out).transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.o_proj(out)
