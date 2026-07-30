"""Kimi LatentMoE / KimiLM tests."""

import torch

from smollms.variants.kimi import KimiLM, KimiLMConfig
from smollms.variants.kimi.latent_moe import LatentMoE


def test_latent_moe_shape_and_aux():
    moe = LatentMoE(d_model=32, d_latent=8, n_experts=4, n_active=2, n_shared=1)
    moe.train()
    x = torch.randn(2, 5, 32)
    y = moe(x)
    assert y.shape == x.shape
    assert moe.last_aux_loss is not None
    assert moe.last_aux_loss.ndim == 0


def test_kimi_lm_forward_and_loss_drops():
    torch.manual_seed(0)
    cfg = KimiLMConfig(
        vocab_size=20,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        n_experts=4,
        n_active=2,
        n_shared=1,
        max_seq_len=32,
    )
    model = KimiLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, 20, (4, 12))
    y = torch.randint(0, 20, (4, 12))

    losses = []
    for _ in range(12):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]


def test_kimi_generate():
    cfg = KimiLMConfig(vocab_size=15, d_model=32, n_layers=1, n_heads=2, max_seq_len=16)
    model = KimiLM(cfg)
    out = model.generate(torch.randint(0, 15, (1, 3)), max_new_tokens=4)
    assert out.shape == (1, 7)
