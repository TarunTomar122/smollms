"""Rotary Position Embeddings (RoPE).

Problem
-------
Token embeddings do not know *where* a token sits in the sequence. Older models
added a position embedding table. RoPE instead **rotates** query/key vectors by
an angle that depends on position, so attention scores become a function of
*relative* distance.

Idea (one head dimension pair)
------------------------------
For position ``t`` and frequency ``θ``:

    rotate(q, t) =  [ q0 ] [ cos(tθ)  -sin(tθ) ]   = complex multiply by e^{i t θ}
                    [ q1 ] [ sin(tθ)   cos(tθ) ]

Even/odd feature pairs are treated as 2D coordinates and spun by ``t * θ``.
Different pairs use different base frequencies (long and short range).

We apply RoPE to **Q and K only**, not V. Values stay unrotated; position
enters only through the QK score geometry.

Shapes
------
cos, sin: (T, head_dim)   or broadcastable to q/k
q, k:     (B, n_heads, T, head_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for positions ``0 .. seq_len-1``.

    Returns:
        cos, sin each of shape ``(seq_len, head_dim)``, ready to multiply with q/k
        after the standard "interleaved pair" rearrange (see ``apply_rope``).
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head_dim must be even, got {head_dim}")

    # θ_i = base^{-2i/d} for i = 0 .. d/2-1
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, device=device, dtype=torch.float32) * 2.0 / head_dim)
    )
    # positions t = 0 .. T-1
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    # outer product: freqs[t, i] = t * θ_i   shape (T, half)
    freqs = torch.outer(t, inv_freq)
    # duplicate each frequency so we can multiply elementwise with (x_even, x_odd)
    # layout: [θ0, θ0, θ1, θ1, ...] after we interleave in apply — easier: cat
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
    cos = emb.cos().to(dtype=dtype)
    sin = emb.sin().to(dtype=dtype)
    return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to q or k.

    Args:
        x: (B, n_heads, T, head_dim)
        cos, sin: (T, head_dim) or (1, 1, T, head_dim)

    Returns:
        Rotated tensor, same shape as ``x``.
    """
    # Broadcast cos/sin to (1, 1, T, head_dim)
    if cos.ndim == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    # match dtype (e.g. model in bfloat16, cache in float32)
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    # For the standard pair layout we need cos/sin aligned with (even, odd).
    # build_rope_cache stores [θ,θ] as [cos0, cos1, ...] for cat([f,f]) which
    # pairs dims 0..half with half..dim. That matches the *chunk* formulation:
    #   x1, x2 = x[..., :half], x[..., half:]
    # We use rotate_half on the interleaved layout below after rearranging.

    # Convert to "half-and-half" layout expected by cat([freqs, freqs]):
    # first half dims rotate with second half.
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    # rotation: (x1, x2) → (x1 cos - x2 sin, x1 sin + x2 cos)
    # Using cat form with matching cos/sin halves:
    cos1, cos2 = cos[..., :half], cos[..., half:]
    sin1, sin2 = sin[..., :half], sin[..., half:]
    # cos1 should equal cos2 (we duplicated freqs); same for sin
    o1 = x1 * cos1 - x2 * sin1
    o2 = x1 * sin2 + x2 * cos2
    return torch.cat([o1, o2], dim=-1)


class RotaryEmbedding(nn.Module):
    """Caches cos/sin up to ``max_seq_len`` and slices per forward length."""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        cos, sin = build_rope_cache(max_seq_len, head_dim, base=base)
        # register as buffers: saved with state_dict, not trained
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds RoPE max_seq_len {self.max_seq_len}"
            )
        return self.cos[:seq_len], self.sin[:seq_len]
