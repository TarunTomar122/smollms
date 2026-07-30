"""SwiGLU FFN tests."""

import torch

from smollms.atoms.ffn import SwiGLUFFN, swiglu_hidden_dim


def test_ffn_shape():
    B, T, D, H = 2, 5, 32, 64
    ffn = SwiGLUFFN(D, H)
    y = ffn(torch.randn(B, T, D))
    assert y.shape == (B, T, D)


def test_ffn_position_independent():
    """Same vector at two positions → same FFN output (no cross-token mix)."""
    torch.manual_seed(0)
    ffn = SwiGLUFFN(16, 32)
    ffn.eval()
    v = torch.randn(16)
    x = torch.stack([v, v], dim=0).unsqueeze(0)  # (1, 2, 16)
    y = ffn(x)
    assert torch.allclose(y[0, 0], y[0, 1], atol=1e-5)


def test_swiglu_hidden_dim_rounding():
    # 8/3 * 32 ≈ 85.33 → round up to multiple of 8 → 88
    assert swiglu_hidden_dim(32, mult=8 / 3, multiple_of=8) == 88


def test_param_count():
    D, H = 16, 48
    ffn = SwiGLUFFN(D, H, bias=False)
    # 3 matrices: D*H + D*H + H*D
    assert sum(p.numel() for p in ffn.parameters()) == 3 * D * H
