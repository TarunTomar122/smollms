"""Token embeddings — how discrete ids enter the residual stream.

A language model never sees characters or words directly. A tokenizer maps
text → integer ids in ``{0, 1, ..., vocab_size - 1}``. The embedding layer is
just a lookup table:

    id  i  →  row  E[i]  of shape (d_model,)

So a batch of token ids becomes the initial residual stream:

    input_ids:  (batch, seq)           int64
    embed:      (batch, seq, d_model)  float

That tensor is exactly the stream RMSNorm / attention / FFN will read and write.

What this is NOT
----------------
- Not "understanding" yet — rows start random (or from pretrained weights).
- Not positions — pure token embed has no notion of order. We will add
  position via **RoPE inside attention** (Qwen / Llama style), not by adding
  a second absolute position embedding table (old GPT-2 style).

Scale note
----------
Some codebases multiply embeddings by ``sqrt(d_model)`` (original Transformer).
Modern decoder LLMs usually do **not**; they rely on RMSNorm right after.
We follow the modern convention: raw lookup, no extra scale.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Lookup table: token id → residual-stream vector.

    Args:
        vocab_size: number of distinct token ids (including specials).
        d_model: width of the residual stream (a.k.a. hidden size).
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        if d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {d_model}")
        self.vocab_size = vocab_size
        self.d_model = d_model
        # nn.Embedding is literally a matrix of shape (vocab_size, d_model)
        # plus a fast gather for integer indices.
        self.table = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token ids.

        Args:
            input_ids: integer tensor of shape (batch, seq) or (seq,).
                Values must be in ``[0, vocab_size)``.

        Returns:
            Float tensor of shape ``(*input_ids.shape, d_model)``.
        """
        if input_ids.dtype not in (torch.int32, torch.int64, torch.long):
            raise TypeError(
                f"input_ids must be integer dtype, got {input_ids.dtype}"
            )
        return self.table(input_ids)

    @property
    def weight(self) -> torch.Tensor:
        """The underlying (vocab_size, d_model) parameter matrix."""
        return self.table.weight

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, d_model={self.d_model}"
