"""Small V4-inspired decoder: local compressed attention plus dense SwiGLU."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from smollms.atoms.embed import TokenEmbedding
from smollms.atoms.ffn import SwiGLUFFN, swiglu_hidden_dim
from smollms.atoms.norm import RMSNorm
from smollms.atoms.residual import residual_add
from smollms.variants.deepseek_v4.attention import LocalCompressedAttention
from smollms.variants.deepseek_v4.moe import V4MoE


@dataclass
class V4ToyLMConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    local_window: int = 16
    compression_ratio: int = 4
    moe_mode: str = "dense"
    n_hash_moe_layers: int = 1
    n_experts: int = 4
    n_active: int = 2
    n_shared: int = 1
    swiglu_limit: float = 10.0
    max_seq_len: int = 256
    rope_base: float = 10000.0
    qk_norm: bool = True
    dropout: float = 0.0
    tie_weights: bool = True
    bias: bool = False

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.local_window < 1 or self.compression_ratio < 1:
            raise ValueError("local_window and compression_ratio must be >= 1")
        if self.moe_mode not in ("dense", "mixed"):
            raise ValueError("moe_mode must be 'dense' or 'mixed'")
        if not 0 <= self.n_hash_moe_layers <= self.n_layers:
            raise ValueError("n_hash_moe_layers must be between 0 and n_layers")


class V4ToyBlock(nn.Module):
    def __init__(self, config: V4ToyLMConfig, mlp_kind: str) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = LocalCompressedAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            local_window=config.local_window,
            compression_ratio=config.compression_ratio,
            max_seq_len=config.max_seq_len,
            rope_base=config.rope_base,
            qk_norm=config.qk_norm,
            dropout=config.dropout,
            bias=config.bias,
        )
        self.mlp_kind = mlp_kind
        self.ffn_norm = RMSNorm(config.d_model) if mlp_kind == "dense" else None
        self.ffn = (
            SwiGLUFFN(
                d_model=config.d_model,
                hidden_dim=swiglu_hidden_dim(config.d_model),
                dropout=config.dropout,
                bias=config.bias,
            )
            if mlp_kind == "dense"
            else None
        )
        self.moe = (
            None
            if mlp_kind == "dense"
            else V4MoE(
                d_model=config.d_model,
                n_experts=config.n_experts,
                n_active=config.n_active,
                n_shared=config.n_shared,
                swiglu_limit=config.swiglu_limit,
                bias=config.bias,
            )
        )

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        x = residual_add(x, self.attn(self.attn_norm(x)))
        if self.moe is not None:
            return residual_add(x, self.moe(x, input_ids if self.mlp_kind == "hash_moe" else None))
        return residual_add(x, self.ffn(self.ffn_norm(x)))


class V4ToyLM(nn.Module):
    def __init__(self, config: V4ToyLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = TokenEmbedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                V4ToyBlock(
                    config,
                    "dense"
                    if config.moe_mode == "dense"
                    else "hash_moe" if i < config.n_hash_moe_layers else "moe",
                )
                for i in range(config.n_layers)
            ]
        )
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
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(f"sequence length {input_ids.size(1)} > max_seq_len {self.config.max_seq_len}")
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, input_ids)
        logits = self.lm_head(self.final_norm(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(input_ids[:, -self.config.max_seq_len :])
            probs = F.softmax(logits[:, -1] / max(temperature, 1e-6), dim=-1)
            input_ids = torch.cat((input_ids, torch.multinomial(probs, 1)), dim=1)
        return input_ids

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_count_unique(self) -> int:
        return sum(p.numel() for p in set(self.parameters()))
