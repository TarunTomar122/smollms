"""RoPE unit tests."""

import torch

from smollms.atoms.rope import RotaryEmbedding, apply_rope, build_rope_cache


def test_build_rope_cache_shapes():
    cos, sin = build_rope_cache(seq_len=16, head_dim=8)
    assert cos.shape == (16, 8)
    assert sin.shape == (16, 8)


def test_rope_rejects_odd_head_dim():
    try:
        build_rope_cache(8, head_dim=7)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_apply_rope_preserves_shape():
    B, H, T, D = 2, 4, 10, 16
    q = torch.randn(B, H, T, D)
    cos, sin = build_rope_cache(T, D)
    out = apply_rope(q, cos, sin)
    assert out.shape == q.shape


def test_rope_is_norm_preserving():
    """Rotation should not change vector L2 norm (per position/head)."""
    torch.manual_seed(0)
    q = torch.randn(1, 2, 5, 32)
    cos, sin = build_rope_cache(5, 32)
    out = apply_rope(q, cos, sin)
    n0 = q.norm(dim=-1)
    n1 = out.norm(dim=-1)
    assert torch.allclose(n0, n1, atol=1e-5)


def test_position_zero_is_near_identity_direction():
    """At t=0, angles are 0 → rotation is identity."""
    q = torch.randn(1, 1, 4, 8)
    cos, sin = build_rope_cache(4, 8)
    # only position 0
    q0 = q[:, :, :1, :]
    out0 = apply_rope(q0, cos[:1], sin[:1])
    assert torch.allclose(out0, q0, atol=1e-5)


def test_rotary_module_slices():
    rope = RotaryEmbedding(head_dim=8, max_seq_len=32)
    cos, sin = rope(10)
    assert cos.shape == (10, 8)
    assert sin.shape == (10, 8)


def test_different_positions_change_vector():
    torch.manual_seed(1)
    q = torch.randn(1, 1, 2, 16)
    # same content at two positions
    q = q[:, :, :1, :].expand(1, 1, 2, 16).contiguous()
    cos, sin = build_rope_cache(2, 16)
    out = apply_rope(q, cos, sin)
    # after RoPE, position 0 and 1 should differ
    assert not torch.allclose(out[:, :, 0, :], out[:, :, 1, :], atol=1e-5)
