"""TinyLM unit tests."""

import torch

from smollms.model import TinyLM, TinyLMConfig


def _small_config(**kwargs) -> TinyLMConfig:
    defaults = dict(
        vocab_size=20,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        qk_norm=True,
        tie_weights=True,
    )
    defaults.update(kwargs)
    return TinyLMConfig(**defaults)


def test_forward_shapes_and_loss():
    cfg = _small_config()
    model = TinyLM(cfg)
    B, T = 3, 8
    x = torch.randint(0, cfg.vocab_size, (B, T))
    y = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(x, y)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss is not None and loss.ndim == 0
    assert loss.item() > 0


def test_forward_without_targets():
    cfg = _small_config()
    model = TinyLM(cfg)
    logits, loss = model(torch.randint(0, 20, (2, 5)))
    assert logits.shape == (2, 5, 20)
    assert loss is None


def test_weight_tying_shares_storage():
    cfg = _small_config(tie_weights=True)
    model = TinyLM(cfg)
    assert model.lm_head.weight.data_ptr() == model.embed.table.weight.data_ptr()


def test_untied_has_separate_weights():
    cfg = _small_config(tie_weights=False)
    model = TinyLM(cfg)
    assert model.lm_head.weight.data_ptr() != model.embed.table.weight.data_ptr()


def test_generate_extends_sequence():
    cfg = _small_config()
    model = TinyLM(cfg)
    prompt = torch.randint(0, 20, (1, 3))
    out = model.generate(prompt, max_new_tokens=5, temperature=1.0)
    assert out.shape == (1, 8)


def test_loss_decreases_few_steps():
    """Smoke: a few Adam steps should reduce loss on a fixed batch."""
    torch.manual_seed(0)
    cfg = _small_config(n_layers=2, d_model=32)
    model = TinyLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, cfg.vocab_size, (8, 16))
    y = torch.randint(0, cfg.vocab_size, (8, 16))

    def batch_loss() -> float:
        _, loss = model(x, y)
        return loss.item()

    losses = []
    for _ in range(15):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
