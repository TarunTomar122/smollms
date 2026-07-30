"""Phase 0.5: token embedding tests."""

import torch

from smollms.atoms import RMSNorm, TokenEmbedding, residual_add


def test_embed_shape():
    emb = TokenEmbedding(vocab_size=100, d_model=32)
    ids = torch.randint(0, 100, (4, 16))
    out = emb(ids)
    assert out.shape == (4, 16, 32)


def test_embed_1d_ids():
    emb = TokenEmbedding(20, 8)
    ids = torch.arange(5)
    assert emb(ids).shape == (5, 8)


def test_same_id_same_vector():
    torch.manual_seed(0)
    emb = TokenEmbedding(10, 16)
    ids = torch.tensor([[3, 7, 3]])
    out = emb(ids)
    assert torch.equal(out[0, 0], out[0, 2])
    assert not torch.equal(out[0, 0], out[0, 1])


def test_embedding_is_table_lookup():
    emb = TokenEmbedding(5, 4)
    with torch.no_grad():
        emb.table.weight.copy_(torch.arange(20, dtype=torch.float).view(5, 4))
    ids = torch.tensor([[2]])
    # row 2 of the table
    expected = emb.weight[2]
    assert torch.equal(emb(ids)[0, 0], expected)


def test_param_count():
    emb = TokenEmbedding(50, 16)
    n = sum(p.numel() for p in emb.parameters())
    assert n == 50 * 16


def test_rejects_float_ids():
    emb = TokenEmbedding(10, 8)
    try:
        emb(torch.randn(2, 3))
        assert False, "expected TypeError"
    except TypeError:
        pass


def test_embed_then_pre_norm_residual_pipeline():
    """First two atoms chained the way a real model starts."""
    B, T, V, D = 2, 6, 40, 24
    emb = TokenEmbedding(V, D)
    norm = RMSNorm(D)
    fake_ffn = torch.nn.Linear(D, D)

    ids = torch.randint(0, V, (B, T))
    stream = emb(ids)
    update = fake_ffn(norm(stream))
    stream = residual_add(stream, update)

    assert stream.shape == (B, T, D)
