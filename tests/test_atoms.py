"""Phase 0: RMSNorm + residual stream shape / smoke tests."""

import torch

from smollms.atoms import RMSNorm, residual_add


def test_rmsnorm_preserves_shape():
    B, T, D = 3, 5, 32
    x = torch.randn(B, T, D)
    y = RMSNorm(D)(x)
    assert y.shape == (B, T, D)


def test_rmsnorm_unit_rms_roughly():
    """After norm (before weight), RMS of features ≈ 1 per token."""
    torch.manual_seed(0)
    D = 64
    x = torch.randn(4, 7, D) * 10.0  # large scale on purpose
    norm = RMSNorm(D)
    # weight starts at ones, so output RMS should be ~1
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_rmsnorm_weight_scales_output():
    D = 16
    x = torch.randn(2, 3, D)
    norm = RMSNorm(D)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    y = norm(x)
    # With γ=2, RMS should be ~2
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, 2.0), atol=1e-5)


def test_residual_add_matches_plus():
    x = torch.randn(2, 4, 8)
    u = torch.randn(2, 4, 8)
    assert torch.equal(residual_add(x, u), x + u)


def test_residual_add_rejects_shape_mismatch():
    x = torch.randn(2, 4, 8)
    u = torch.randn(2, 4, 7)
    try:
        residual_add(x, u)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "matching shapes" in str(e)


def test_pre_norm_residual_pattern_shapes():
    """The pattern every block will use: x + f(norm(x))."""
    B, T, D = 2, 6, 24
    x = torch.randn(B, T, D)
    norm = RMSNorm(D)

    # stand-in for attention or FFN: linear map, same dim
    fake_sublayer = torch.nn.Linear(D, D)
    update = fake_sublayer(norm(x))
    out = residual_add(x, update)

    assert out.shape == (B, T, D)
    # stream is not *replaced* — identity path still there
    assert not torch.equal(out, update)
