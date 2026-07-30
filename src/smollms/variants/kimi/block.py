"""Kimi-style block.

Step 1: full causal attention + LatentMoE
Step 2: full attention, minimal KDA, or minimal Gated MLA + LatentMoE
        (AttnRes is applied at model level across blocks)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from smollms.atoms.attention import CausalSelfAttention
from smollms.atoms.norm import RMSNorm
from smollms.atoms.residual import residual_add
from smollms.variants.kimi.latent_moe import LatentMoE
from smollms.variants.kimi.linear_attn import KimiDeltaAttention
from smollms.variants.kimi.mla import GatedMultiHeadLatentAttention


class KimiBlock(nn.Module):
    """Pre-norm token mixer (full attention, KDA, or MLA) + LatentMoE."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        d_latent: int | None = None,
        n_experts: int = 4,
        n_active: int = 2,
        n_shared: int = 1,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        qk_norm: bool = True,
        dropout: float = 0.0,
        aux_loss_weight: float = 0.01,
        bias: bool = False,
        attn_type: str = "full",
    ) -> None:
        super().__init__()
        if attn_type not in ("full", "kda", "mla"):
            raise ValueError(f"attn_type must be 'full', 'kda', or 'mla', got {attn_type!r}")
        self.attn_type = attn_type
        self.attn_norm = RMSNorm(d_model)

        if attn_type == "full":
            self.attn: nn.Module = CausalSelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
                qk_norm=qk_norm,
                dropout=dropout,
                bias=bias,
            )
        elif attn_type == "kda":
            # KDA path: one recurrent state matrix per head.
            self.attn = KimiDeltaAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                bias=bias,
            )
        else:
            self.attn = GatedMultiHeadLatentAttention(
                d_model=d_model,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
                qk_norm=qk_norm,
                dropout=dropout,
                bias=bias,
            )

        self.moe = LatentMoE(
            d_model=d_model,
            d_latent=d_latent,
            n_experts=n_experts,
            n_active=n_active,
            n_shared=n_shared,
            aux_loss_weight=aux_loss_weight,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = residual_add(x, self.attn(self.attn_norm(x)))
        x = residual_add(x, self.moe(x))
        return x

    def aux_loss(self) -> torch.Tensor:
        if self.moe.last_aux_loss is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.moe.last_aux_loss
