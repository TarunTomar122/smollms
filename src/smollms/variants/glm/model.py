"""Minimal GLM-style language model: DSA or dense attention plus routed MoE."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from smollms.atoms.embed import TokenEmbedding
from smollms.atoms.norm import RMSNorm
from smollms.variants.glm.block import GLMBlock


@dataclass
class GLMLMConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    attention_kind: str = "sparse"
    sparse_top_k: int = 8
    n_dense_layers: int = 1
    n_experts: int = 4
    n_active: int = 2
    n_shared: int = 1
    max_seq_len: int = 256
    rope_base: float = 10000.0
    qk_norm: bool = True
    dropout: float = 0.0
    tie_weights: bool = True
    bias: bool = False

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.sparse_top_k < 1:
            raise ValueError("sparse_top_k must be >= 1")
        if self.attention_kind not in ("sparse", "dense"):
            raise ValueError("attention_kind must be 'sparse' or 'dense'")
        if not 0 <= self.n_dense_layers <= self.n_layers:
            raise ValueError("n_dense_layers must be between 0 and n_layers")


class GLMLM(nn.Module):
    """Decoder-only toy with DSA in every layer and a routed MoE FFN."""

    def __init__(self, config: GLMLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = TokenEmbedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                GLMBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    attention_kind=config.attention_kind,
                    sparse_top_k=config.sparse_top_k,
                    n_experts=config.n_experts,
                    n_active=config.n_active,
                    n_shared=config.n_shared,
                    use_moe=i >= config.n_dense_layers,
                    max_seq_len=config.max_seq_len,
                    rope_base=config.rope_base,
                    qk_norm=config.qk_norm,
                    dropout=config.dropout,
                    bias=config.bias,
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
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.size(1)} > max_seq_len {self.config.max_seq_len}"
            )
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(input_ids[:, -self.config.max_seq_len :])
            probs = F.softmax(logits[:, -1] / max(temperature, 1e-6), dim=-1)
            input_ids = torch.cat([input_ids, torch.multinomial(probs, 1)], dim=1)
        return input_ids

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_count_unique(self) -> int:
        return sum(p.numel() for p in set(self.parameters()))
