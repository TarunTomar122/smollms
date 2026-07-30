"""GLM toy: DSA + routed MoE tests."""

import torch

from smollms.atoms.attention import CausalSelfAttention
from smollms.variants.glm import GLMLM, GLMLMConfig
from smollms.variants.glm.moe import GLMMoE


def test_glm_uses_dense_prefix_and_aux_free_sigmoid_router():
    model = GLMLM(
        GLMLMConfig(vocab_size=20, d_model=32, n_layers=4, n_heads=4, n_dense_layers=1)
    )
    assert not model.blocks[0].uses_moe
    assert all(block.uses_moe for block in model.blocks[1:])

    moe = GLMMoE(d_model=8, n_experts=4, n_active=2)
    with torch.no_grad():
        moe.router.weight.zero_()
    moe.train()
    _ = moe(torch.randn(2, 3, 8))
    assert moe.last_aux_loss is not None
    assert moe.last_aux_loss.item() == 0.0
    assert not torch.equal(moe.expert_correction_bias, torch.zeros(4))


def test_glm_dense_attention_ablation_keeps_the_corrected_moe():
    model = GLMLM(
        GLMLMConfig(
            vocab_size=20,
            d_model=32,
            n_layers=4,
            n_heads=4,
            n_dense_layers=1,
            attention_kind="dense",
        )
    )

    assert all(isinstance(block.attn, CausalSelfAttention) for block in model.blocks)
    assert not model.blocks[0].uses_moe
    assert all(block.uses_moe for block in model.blocks[1:])


def test_glm_forward_loss_drops_and_generates():
    torch.manual_seed(0)
    cfg = GLMLMConfig(
        vocab_size=20,
        d_model=32,
        n_layers=2,
        n_heads=4,
        sparse_top_k=4,
        n_experts=4,
        n_active=2,
        n_shared=1,
        max_seq_len=32,
    )
    model = GLMLM(cfg)
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
    out = model.generate(x[:1, :3], max_new_tokens=4)
    assert out.shape == (1, 7)
