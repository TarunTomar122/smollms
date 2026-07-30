"""Tiny dense decoder LM — Qwen/Llama-style block stack.

    token ids
        → TokenEmbedding
        → TransformerBlock × n_layers
        → RMSNorm
        → lm_head  (logits over vocab)

Default: weight-tied lm_head (uses embedding table as output projection).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from smollms.atoms.embed import TokenEmbedding
from smollms.atoms.norm import RMSNorm
from smollms.blocks.transformer import TransformerBlock


@dataclass
class TinyLMConfig:
    """Hyperparameters for a toy dense LM."""

    vocab_size: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int | None = None  # None → full MHA
    hidden_dim: int | None = None  # None → SwiGLU heuristic
    max_seq_len: int = 256
    rope_base: float = 10000.0
    qk_norm: bool = True
    dropout: float = 0.0
    tie_weights: bool = True
    bias: bool = False

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")


class TinyLM(nn.Module):
    """Small causal language model built from our atoms + TransformerBlock."""

    def __init__(self, config: TinyLMConfig) -> None:
        super().__init__()
        self.config = config

        self.embed = TokenEmbedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    n_kv_heads=config.n_kv_heads,
                    hidden_dim=config.hidden_dim,
                    max_seq_len=config.max_seq_len,
                    rope_base=config.rope_base,
                    qk_norm=config.qk_norm,
                    dropout=config.dropout,
                    bias=config.bias,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = RMSNorm(config.d_model)
        # Always create lm_head; may share weight storage with embed if tied.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            # Same storage: embed.table.weight is (V, D); lm_head.weight is (V, D).
            self.lm_head.weight = self.embed.table.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Simple normal init; good enough for toys."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Skip if this is the tied lm_head (embedding already inited).
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
        """
        Args:
            input_ids: (B, T) token ids.
            targets: optional (B, T) next-token ids for loss.

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar cross-entropy, or None if targets is None.
        """
        b, t = input_ids.shape
        if t > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {t} > max_seq_len {self.config.max_seq_len}"
            )

        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten for cross_entropy: (B*T, V) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Greedy/sample autoregressive decode (toy helper).

        Args:
            input_ids: (B, T) prompt.
            max_new_tokens: how many tokens to append.
            temperature: softmax temperature (1.0 = plain sampling).

        Returns:
            (B, T + max_new_tokens) full sequence.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # crop context to max_seq_len
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
        """Count unique parameters (weight tying shares storage)."""
        return sum(p.numel() for p in set(self.parameters()))
