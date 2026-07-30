"""Char data tests."""

import torch

from smollms.data.char_data import (
    CharDataset,
    CharTokenizer,
    load_text,
    prepare_char_data,
)


def test_tokenizer_roundtrip():
    text = "hello world"
    tok = CharTokenizer(text)
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert tok.vocab_size == len(set(text))


def test_dataset_shift():
    data = torch.arange(20)
    ds = CharDataset(data, block_size=5)
    x, y = ds[0]
    assert x.tolist() == [0, 1, 2, 3, 4]
    assert y.tolist() == [1, 2, 3, 4, 5]


def test_prepare_char_data():
    text = load_text(None)
    tok, train_ds, val_ds = prepare_char_data(text, block_size=32)
    assert tok.vocab_size > 10
    assert len(train_ds) > 0
    x, y = train_ds[0]
    assert x.shape == (32,)
    assert y.shape == (32,)
