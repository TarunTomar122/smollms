"""Causal attention tests."""

import torch

from smollms.atoms import (
    CausalSelfAttention,
    RMSNorm,
    TokenEmbedding,
    residual_add,
)


def test_attention_shape_mha():
    B, T, D, H = 2, 7, 32, 4
    attn = CausalSelfAttention(D, n_heads=H)
    x = torch.randn(B, T, D)
    y = attn(x)
    assert y.shape == (B, T, D)


def test_attention_shape_gqa():
    B, T, D = 2, 6, 32
    attn = CausalSelfAttention(D, n_heads=4, n_kv_heads=2, qk_norm=True)
    y = attn(torch.randn(B, T, D))
    assert y.shape == (B, T, D)


def test_causal_no_future_leak_gradient():
    """output[t] must not depend on input[t+1].

    If we change a future token and output at t is unchanged, mask works.
    """
    torch.manual_seed(0)
    D, H, T = 16, 2, 5
    attn = CausalSelfAttention(D, n_heads=H, dropout=0.0)
    attn.eval()

    x = torch.randn(1, T, D)
    y = attn(x)

    x2 = x.clone()
    x2[0, -1] = x2[0, -1] + 10.0  # perturb last position only
    y2 = attn(x2)

    # positions 0 .. T-2 should match; last may differ
    assert torch.allclose(y[0, :-1], y2[0, :-1], atol=1e-5)
    # last position is allowed to change
    assert not torch.allclose(y[0, -1], y2[0, -1], atol=1e-3)


def test_qk_norm_flag_creates_params():
    a0 = CausalSelfAttention(32, n_heads=4, qk_norm=False)
    a1 = CausalSelfAttention(32, n_heads=4, qk_norm=True)
    n0 = sum(p.numel() for p in a0.parameters())
    n1 = sum(p.numel() for p in a1.parameters())
    # two RMSNorm weights of size head_dim each
    head_dim = 32 // 4
    assert n1 - n0 == 2 * head_dim


def test_embed_norm_attn_residual_pipeline():
    B, T, V, D = 2, 8, 40, 32
    emb = TokenEmbedding(V, D)
    norm = RMSNorm(D)
    attn = CausalSelfAttention(D, n_heads=4, n_kv_heads=2, qk_norm=True, max_seq_len=128)

    ids = torch.randint(0, V, (B, T))
    stream = emb(ids)
    stream = residual_add(stream, attn(norm(stream)))
    assert stream.shape == (B, T, D)


def test_invalid_gqa_config():
    try:
        CausalSelfAttention(32, n_heads=4, n_kv_heads=3)
        assert False, "expected ValueError"
    except ValueError:
        pass
