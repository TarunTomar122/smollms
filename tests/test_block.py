"""Full transformer block tests."""

import torch

from smollms.atoms import TokenEmbedding
from smollms.blocks import TransformerBlock


def test_block_shape():
    B, T, D = 2, 9, 32
    block = TransformerBlock(d_model=D, n_heads=4, n_kv_heads=2)
    y = block(torch.randn(B, T, D))
    assert y.shape == (B, T, D)


def test_block_stack():
    B, T, V, D = 2, 6, 30, 32
    emb = TokenEmbedding(V, D)
    blocks = torch.nn.ModuleList(
        [TransformerBlock(D, n_heads=4, n_kv_heads=2, max_seq_len=64) for _ in range(3)]
    )
    x = emb(torch.randint(0, V, (B, T)))
    for blk in blocks:
        x = blk(x)
    assert x.shape == (B, T, D)


def test_block_grad_flows():
    """Identity residual path should let grads reach the input embed path."""
    D, V = 16, 20
    emb = TokenEmbedding(V, D)
    block = TransformerBlock(D, n_heads=2, qk_norm=True, max_seq_len=32)
    ids = torch.randint(0, V, (2, 5))
    x = emb(ids)
    loss = block(x).sum()
    loss.backward()
    assert emb.table.weight.grad is not None
    assert emb.table.weight.grad.abs().sum() > 0


def test_qk_norm_default_true():
    block = TransformerBlock(32, n_heads=4)
    assert block.attn.qk_norm_enabled is True
