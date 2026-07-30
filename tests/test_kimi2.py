"""Kimi step-2: KDA, Gated MLA, and AttnRes tests."""

import torch

from smollms.variants.kimi import KimiLM, KimiLMConfig
from smollms.variants.kimi.attn_res import AttentionResidual
from smollms.variants.kimi.linear_attn import KimiDeltaAttention, kda_recurrence
from smollms.variants.kimi.mla import GatedMultiHeadLatentAttention
from smollms.variants.kimi.model import _hybrid_attn_types


def test_hybrid_pattern_3l1f():
    assert _hybrid_attn_types(4, "3L1F") == ["kda", "kda", "kda", "mla"]
    assert _hybrid_attn_types(5, "3L1F") == ["kda", "kda", "kda", "mla", "kda"]


def test_kda_delta_rule_overwrites_an_existing_key():
    q = k = torch.ones(1, 1, 2, 1)
    v = torch.tensor([[[[1.0], [3.0]]]])
    decay = torch.ones(1, 1, 2, 1)
    beta = torch.ones(1, 1, 2, 1)

    out = kda_recurrence(q, k, v, decay, beta)

    assert torch.allclose(out.squeeze(), torch.tensor([1.0, 3.0]))


def test_kda_shape_and_causal_grad():
    torch.manual_seed(0)
    attn = KimiDeltaAttention(d_model=32, n_heads=4)
    attn.eval()
    x = torch.randn(1, 6, 32)
    y = attn(x)
    assert y.shape == x.shape

    # future token should not affect earlier outputs
    x2 = x.clone()
    x2[0, -1] = x2[0, -1] + 5.0
    y2 = attn(x2)
    assert torch.allclose(y[0, :-1], y2[0, :-1], atol=1e-4)


def test_gated_mla_shape_and_causality():
    torch.manual_seed(0)
    attn = GatedMultiHeadLatentAttention(d_model=32, n_heads=4)
    attn.eval()
    x = torch.randn(1, 6, 32)
    y = attn(x)
    assert y.shape == x.shape

    x2 = x.clone()
    x2[0, -1] = x2[0, -1] + 5.0
    y2 = attn(x2)
    assert torch.allclose(y[0, :-1], y2[0, :-1], atol=1e-4)


def test_attn_res_chooses_history_using_current_stream():
    ar = AttentionResidual(2, depth=2)
    with torch.no_grad():
        ar.out_proj.weight.copy_(torch.eye(2))
        ar.gate_proj.weight.zero_()  # sigmoid(0) = 0.5
        ar.gate_bias.zero_()

    older = torch.tensor([[[2.0, 0.0]]])
    newer = torch.tensor([[[0.0, 2.0]]])
    x0 = torch.tensor([[[1.0, 0.0]]])
    x1 = torch.tensor([[[0.0, 1.0]]])

    add0 = ar(x0, [older, newer]) - x0
    add1 = ar(x1, [older, newer]) - x1

    assert add0[0, 0, 0] > add0[0, 0, 1]  # chooses older [2, 0]
    assert add1[0, 0, 1] > add1[0, 0, 0]  # chooses newer [0, 2]


def test_kimi2_forward_loss_drops():
    torch.manual_seed(0)
    cfg = KimiLMConfig(
        vocab_size=20,
        d_model=32,
        n_layers=4,
        n_heads=4,
        max_seq_len=32,
        kimi_step=2,
        hybrid_pattern="3L1F",
        use_attn_res=True,
        attn_res_depth=2,
        n_experts=4,
        n_active=2,
    )
    model = KimiLM(cfg)
    assert model.attn_types == ["kda", "kda", "kda", "mla"]
    assert isinstance(model.blocks[-1].attn, GatedMultiHeadLatentAttention)
    assert model.attn_res is not None

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, 20, (4, 12))
    y = torch.randint(0, 20, (4, 12))
    losses = []
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
