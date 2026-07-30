"""Inference load + generate smoke test."""

import torch

from smollms.infer import decode, encode, generate, load_checkpoint
from smollms.model import TinyLM, TinyLMConfig
from smollms.variants.deepseek_v4 import V4ToyLM, V4ToyLMConfig


def test_save_load_generate(tmp_path):
    cfg = TinyLMConfig(
        vocab_size=10,
        d_model=32,
        n_layers=1,
        n_heads=2,
        max_seq_len=16,
        tie_weights=True,
    )
    model = TinyLM(cfg)
    stoi = {chr(ord("a") + i): i for i in range(10)}
    itos = {i: ch for ch, i in stoi.items()}
    path = tmp_path / "toy.pt"
    torch.save(
        {
            "arch": "dense",
            "model": model.state_dict(),
            "config": cfg,
            "stoi": stoi,
            "itos": itos,
        },
        path,
    )


    loaded, stoi2, itos2 = load_checkpoint(path, torch.device("cpu"))
    ids = encode("ab", stoi2)
    x = torch.tensor([ids], dtype=torch.long)
    out = generate(loaded, x, max_new_tokens=5, temperature=1.0, top_k=5)
    assert out.shape[1] == 2 + 5
    text = decode(out[0], itos2)
    assert len(text) == 7


def test_v4_save_load_generate(tmp_path):
    cfg = V4ToyLMConfig(vocab_size=10, d_model=32, n_layers=1, n_heads=2, max_seq_len=16)
    model = V4ToyLM(cfg)
    stoi = {chr(ord("a") + i): i for i in range(10)}
    itos = {i: ch for ch, i in stoi.items()}
    path = tmp_path / "v4.pt"
    torch.save(
        {"arch": "deepseekv4", "model": model.state_dict(), "config": cfg, "stoi": stoi, "itos": itos},
        path,
    )

    loaded, stoi2, _ = load_checkpoint(path, torch.device("cpu"))
    out = generate(loaded, torch.tensor([encode("ab", stoi2)]), max_new_tokens=3)
    assert out.shape == (1, 5)
