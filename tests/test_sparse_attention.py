import torch

from smollms.atoms.sparse_attention import DeepSeekSparseAttention


def test_dsa_is_causal_and_trains_its_indexer():
    torch.manual_seed(0)
    attn = DeepSeekSparseAttention(d_model=32, n_heads=4, top_k=2)
    attn.eval()
    x = torch.randn(1, 6, 32)
    y = attn(x)
    assert y.shape == x.shape

    assert attn.last_selected_indices is not None
    assert attn.last_selected_valid is not None
    positions = torch.arange(6).view(1, 1, 6, 1).expand_as(attn.last_selected_indices)
    assert torch.all(
        attn.last_selected_indices[attn.last_selected_valid]
        <= positions[attn.last_selected_valid]
    )

    x2 = x.clone()
    x2[0, -1] = x2[0, -1] + 5.0
    y2 = attn(x2)
    assert torch.allclose(y[0, :-1], y2[0, :-1], atol=1e-4)

    y.sum().backward()
    assert attn.index_q_proj.weight.grad is not None
    assert attn.index_q_proj.weight.grad.abs().sum() > 0
