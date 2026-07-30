"""One minimal GLM block: selectable token mixer + routed MoE channel mixer."""

from __future__ import annotations

import torch
import torch.nn as nn

from smollms.atoms.attention import CausalSelfAttention
from smollms.atoms.ffn import SwiGLUFFN, swiglu_hidden_dim
from smollms.atoms.norm import RMSNorm
from smollms.atoms.residual import residual_add
from smollms.atoms.sparse_attention import DeepSeekSparseAttention
from smollms.variants.glm.moe import GLMMoE


class GLMBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attention_kind: str,
        sparse_top_k: int,
        n_experts: int,
        n_active: int,
        n_shared: int,
        use_moe: bool,
        max_seq_len: int,
        rope_base: float,
        qk_norm: bool,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        if attention_kind == "sparse":
            self.attn = DeepSeekSparseAttention(
                d_model=d_model,
                n_heads=n_heads,
                top_k=sparse_top_k,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
                qk_norm=qk_norm,
                dropout=dropout,
                bias=bias,
            )
        elif attention_kind == "dense":
            self.attn = CausalSelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
                qk_norm=qk_norm,
                dropout=dropout,
                bias=bias,
            )
        else:
            raise ValueError(f"unknown attention_kind {attention_kind!r}")
        self.uses_moe = use_moe
        if use_moe:
            self.moe = GLMMoE(
                d_model=d_model,
                n_experts=n_experts,
                n_active=n_active,
                n_shared=n_shared,
                bias=bias,
            )
            self.ffn_norm = None
            self.ffn = None
        else:
            self.moe = None
            self.ffn_norm = RMSNorm(d_model)
            self.ffn = SwiGLUFFN(
                d_model=d_model,
                hidden_dim=swiglu_hidden_dim(d_model),
                dropout=dropout,
                bias=bias,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = residual_add(x, self.attn(self.attn_norm(x)))
        if self.moe is not None:
            return residual_add(x, self.moe(x))
        return residual_add(x, self.ffn(self.ffn_norm(x)))

    def aux_loss(self) -> torch.Tensor:
        if self.moe is None or self.moe.last_aux_loss is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return self.moe.last_aux_loss
