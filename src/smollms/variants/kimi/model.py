"""Kimi-style tiny LM.

Step 1: LatentMoE + full attention every layer
Step 2: + hybrid KDA/Gated MLA + AttnRes across depth
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from smollms.atoms.embed import TokenEmbedding
from smollms.atoms.norm import RMSNorm
from smollms.variants.kimi.attn_res import AttentionResidual
from smollms.variants.kimi.block import KimiBlock


def _hybrid_attn_types(n_layers: int, pattern: str) -> list[str]:
    """Map a pattern string to per-layer attn types.

    Patterns
    --------
    all_full  — step-1 behaviour
    3L1F      — KDA,KDA,KDA,MLA, repeat  (KDA-dominant toy)
    1L1F      — KDA,MLA,KDA,MLA,...
    """
    if pattern == "all_full":
        return ["full"] * n_layers
    if pattern == "3L1F":
        unit = ["kda", "kda", "kda", "mla"]
    elif pattern == "1L1F":
        unit = ["kda", "mla"]
    else:
        raise ValueError(f"unknown hybrid pattern {pattern!r}")
    return [unit[i % len(unit)] for i in range(n_layers)]


@dataclass
class KimiLMConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int | None = None
    d_latent: int | None = None
    n_experts: int = 4
    n_active: int = 2
    n_shared: int = 1
    max_seq_len: int = 256
    rope_base: float = 10000.0
    qk_norm: bool = True
    dropout: float = 0.0
    aux_loss_weight: float = 0.01
    tie_weights: bool = True
    bias: bool = False
    # --- step 2 knobs ---
    kimi_step: int = 1
    hybrid_pattern: str = "all_full"  # step2 default set by train when step=2
    use_attn_res: bool = False
    attn_res_depth: int = 2

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")


class KimiLM(nn.Module):
    """Kimi-inspired stack: hybrid token mixers + LatentMoE + optional AttnRes."""

    def __init__(self, config: KimiLMConfig) -> None:
        super().__init__()
        self.config = config
        attn_types = _hybrid_attn_types(config.n_layers, config.hybrid_pattern)
        self.attn_types = attn_types

        self.embed = TokenEmbedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                KimiBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    n_kv_heads=config.n_kv_heads,
                    d_latent=config.d_latent,
                    n_experts=config.n_experts,
                    n_active=config.n_active,
                    n_shared=config.n_shared,
                    max_seq_len=config.max_seq_len,
                    rope_base=config.rope_base,
                    qk_norm=config.qk_norm,
                    dropout=config.dropout,
                    aux_loss_weight=config.aux_loss_weight,
                    bias=config.bias,
                    attn_type=attn_types[i],
                )
                for i in range(config.n_layers)
            ]
        )
        self.attn_res: AttentionResidual | None
        if config.use_attn_res:
            self.attn_res = AttentionResidual(
                config.d_model, depth=config.attn_res_depth, bias=config.bias
            )
        else:
            self.attn_res = None

        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.embed.table.weight
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.tie_weights and module is self.lm_head:
                    continue
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t = input_ids.shape
        if t > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {t} > max_seq_len {self.config.max_seq_len}"
            )

        x = self.embed(input_ids)
        aux_total = x.new_zeros(())
        history: list[torch.Tensor] = []

        for block in self.blocks:
            x = block(x)
            aux_total = aux_total + block.aux_loss()
            if self.attn_res is not None:
                x = self.attn_res(x, history)
            # store stream for deeper AttnRes (keep graph — toy depth only)
            history.append(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = ce + aux_total / max(len(self.blocks), 1)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            window = input_ids[:, -self.config.max_seq_len :]
            logits, _ = self(window)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_count_unique(self) -> int:
        return sum(p.numel() for p in set(self.parameters()))
