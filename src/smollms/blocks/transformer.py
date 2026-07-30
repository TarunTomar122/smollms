"""One pre-norm transformer block — dense Qwen / Llama style.

    x ← x + Attention(RMSNorm(x))
    x ← x + SwiGLU(RMSNorm(x))

That is the whole block. Stack N of these between embed and lm_head and you
have a dense decoder LM (minus training tricks).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from smollms.atoms.attention import CausalSelfAttention
from smollms.atoms.ffn import SwiGLUFFN, swiglu_hidden_dim
from smollms.atoms.norm import RMSNorm
from smollms.atoms.residual import residual_add


class TransformerBlock(nn.Module):
    """Single residual block: pre-norm attention + pre-norm SwiGLU.

    Args:
        d_model: residual stream width.
        n_heads: query heads.
        n_kv_heads: key/value heads (GQA). Default = n_heads (full MHA).
        hidden_dim: FFN intermediate size. Default = swiglu heuristic.
        max_seq_len: RoPE cache length.
        qk_norm: Qwen3-style Q/K RMSNorm.
        dropout: attn + ffn dropout.
        bias: linear biases (default False, modern style).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        hidden_dim: int | None = None,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        qk_norm: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = swiglu_hidden_dim(d_model)

        self.d_model = d_model
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            qk_norm=qk_norm,
            dropout=dropout,
            bias=bias,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(
            d_model=d_model,
            hidden_dim=hidden_dim,
            bias=bias,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) residual stream in.

        Returns:
            (B, T, d_model) residual stream out.
        """
        x = residual_add(x, self.attn(self.attn_norm(x)))
        x = residual_add(x, self.ffn(self.ffn_norm(x)))
        return x
