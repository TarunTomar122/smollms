"""Character-level dataset for tiny LM experiments.

Why chars first?
----------------
- vocab_size is small (~65 for ASCII Shakespeare) → tiny embedding table
- no external tokenizer dependency
- you see next-token learning clearly ("hello" → predict 'l', 'o', ...)

Later we can swap in BPE without changing the model API (still int ids).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

# Tiny bundled fallback so training works offline without a download.
_FALLBACK_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
""".strip()


def load_text(path: str | Path | None = None) -> str:
    """Load a text file, or return a small built-in sample."""
    if path is None:
        return _FALLBACK_TEXT * 50  # repeat so we have enough tokens to train
    path = Path(path)
    return path.read_text(encoding="utf-8")


class CharTokenizer:
    """Map characters ↔ integer ids (sorted unique chars)."""

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)


class CharDataset(Dataset):
    """Sliding windows of fixed block_size for next-token prediction.

    Each item:
        x = tokens[i : i+T]
        y = tokens[i+1 : i+T+1]   # next char at each position
    """

    def __init__(self, data: torch.Tensor, block_size: int) -> None:
        if data.ndim != 1:
            raise ValueError("data must be a 1-D LongTensor of token ids")
        if len(data) < block_size + 1:
            raise ValueError(
                f"need at least block_size+1={block_size + 1} tokens, got {len(data)}"
            )
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def prepare_char_data(
    text: str,
    block_size: int,
    train_frac: float = 0.9,
) -> tuple[CharTokenizer, CharDataset, CharDataset]:
    """Build tokenizer + train/val datasets from raw text."""
    tokenizer = CharTokenizer(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(train_frac * len(ids))
    train_ids, val_ids = ids[:n], ids[n:]
    # if val is too short, just reuse a slice of train for smoke tests
    if len(val_ids) < block_size + 1:
        val_ids = ids[max(0, n - (block_size + 64)) :]
    train_ds = CharDataset(train_ids, block_size)
    val_ds = CharDataset(val_ids, block_size)
    return tokenizer, train_ds, val_ds
