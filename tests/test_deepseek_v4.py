"""DeepSeek V4 step-1 local + compressed-memory attention tests."""

import torch

from smollms.variants.deepseek_v4 import LocalCompressedAttention, V4MoE, V4ToyLM, V4ToyLMConfig


def test_local_compressed_attention_is_causal_and_keeps_boundary_tokens():
    torch.manual_seed(0)
    attn = LocalCompressedAttention(
        d_model=32,
        n_heads=4,
        local_window=4,
        compression_ratio=2,
        max_seq_len=16,
    )
    x = torch.randn(1, 8, 32)
    y = attn(x)

    changed = x.clone()
    changed[:, 5:] += torch.randn_like(changed[:, 5:])
    y_changed = attn(changed)
    assert y.shape == x.shape
    assert torch.allclose(y[:, :5], y_changed[:, :5], atol=1e-5)

    compressed, boundary = attn.memory_masks(seq_len=8, device=x.device)
    # At query 6, positions 3..6 are local. Chunk [0, 1] is compressed and
    # position 2 is retained raw at the local/compressed boundary.
    assert compressed[6, 0]
    assert boundary[6, 2]
    assert not boundary[6, 1]


def test_v4_toy_trains_and_generates():
    torch.manual_seed(0)
    model = V4ToyLM(
        V4ToyLMConfig(vocab_size=20, d_model=32, n_layers=2, n_heads=4, max_seq_len=16)
    )
    x = torch.randint(0, 20, (4, 12))
    y = torch.randint(0, 20, (4, 12))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
    assert model.generate(x[:1, :3], max_new_tokens=4).shape == (1, 7)


def test_v4_mixed_moe_uses_hash_bootstrap_then_sqrtsoftplus_routing():
    model = V4ToyLM(
        V4ToyLMConfig(
            vocab_size=20,
            d_model=32,
            n_layers=4,
            n_heads=4,
            moe_mode="mixed",
            n_hash_moe_layers=1,
        )
    )
    assert model.blocks[0].mlp_kind == "hash_moe"
    assert [block.mlp_kind for block in model.blocks[1:]] == ["moe", "moe", "moe"]

    moe = V4MoE(d_model=8, n_experts=4, n_active=2, n_shared=1)
    with torch.no_grad():
        moe.router.weight.zero_()
    token_ids = torch.tensor([[1, 1, 3]])
    _ = moe(torch.randn(1, 3, 8), token_ids=token_ids)
    assert torch.equal(moe.last_selected_experts[0], moe.last_selected_experts[1])
    assert moe.last_selected_experts[0].tolist() == [1, 2]

    moe.train()
    _ = moe(torch.randn(2, 3, 8))
    assert not torch.equal(moe.expert_correction_bias, torch.zeros(4))
