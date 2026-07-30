"""Generate text from a trained smollms checkpoint (dense or kimi).

Usage (from repo root, venv on):

    python -m smollms.infer \\
      --checkpoint checkpoints/shakespeare_dense.pt \\
      --prompt "To be" \\
      --max-new-tokens 300

    python -m smollms.infer -c checkpoints/kimi_char.pt -p "ROMEO:" -n 200 --temp 0.8
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from smollms.model import TinyLM, TinyLMConfig
from smollms.variants.deepseek_v4 import V4ToyLM, V4ToyLMConfig
from smollms.variants.glm import GLMLM, GLMLMConfig
from smollms.variants.kimi import KimiLM, KimiLMConfig


def _detect_arch(ckpt: dict[str, Any]) -> str:
    if "arch" in ckpt:
        return ckpt["arch"]
    cfg = ckpt["config"]
    if isinstance(cfg, GLMLMConfig) or type(cfg).__name__ == "GLMLMConfig":
        return "glm"
    if isinstance(cfg, V4ToyLMConfig) or type(cfg).__name__ == "V4ToyLMConfig":
        return "deepseekv4"
    if isinstance(cfg, KimiLMConfig) or type(cfg).__name__ == "KimiLMConfig":
        step = getattr(cfg, "kimi_step", 1)
        return "kimi2" if step >= 2 else "kimi"
    return "dense"


def _kimi_config_from_ckpt(cfg: Any) -> KimiLMConfig:
    """Rebuild KimiLMConfig with defaults for fields added in step 2."""
    if isinstance(cfg, KimiLMConfig):
        return cfg
    raw = dict(vars(cfg)) if hasattr(cfg, "__dict__") else dict(cfg)
    # drop unknown keys gently
    allowed = {f.name for f in KimiLMConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    raw = {k: v for k, v in raw.items() if k in allowed}
    return KimiLMConfig(**raw)


def _glm_config_from_ckpt(cfg: Any) -> GLMLMConfig:
    raw = dict(vars(cfg)) if hasattr(cfg, "__dict__") else dict(cfg)
    allowed = {f.name for f in GLMLMConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return GLMLMConfig(**{k: v for k, v in raw.items() if k in allowed})


def _v4_config_from_ckpt(cfg: Any) -> V4ToyLMConfig:
    raw = dict(vars(cfg)) if hasattr(cfg, "__dict__") else dict(cfg)
    allowed = {f.name for f in V4ToyLMConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return V4ToyLMConfig(**{k: v for k, v in raw.items() if k in allowed})


def load_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, int], dict[int, str]]:
    """Rebuild model + char maps from a train.py checkpoint."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    stoi: dict[str, int] = ckpt["stoi"]
    itos: dict[int, str] = {int(k): v for k, v in ckpt["itos"].items()}
    arch = _detect_arch(ckpt)
    config = ckpt["config"]

    if arch in ("kimi", "kimi2"):
        config = _kimi_config_from_ckpt(config)
        model: nn.Module = KimiLM(config).to(device)
    elif arch == "glm":
        config = _glm_config_from_ckpt(config)
        model = GLMLM(config).to(device)
    elif arch == "deepseekv4":
        config = _v4_config_from_ckpt(config)
        model = V4ToyLM(config).to(device)
    else:
        if not isinstance(config, TinyLMConfig):
            config = TinyLMConfig(**vars(config))
        model = TinyLM(config).to(device)

    model.load_state_dict(ckpt["model"])
    # Re-bind after load so embed and lm_head share storage again.
    if getattr(config, "tie_weights", False):
        model.lm_head.weight = model.embed.table.weight  # type: ignore[attr-defined]
    model.eval()
    return model, stoi, itos


def encode(prompt: str, stoi: dict[str, int]) -> list[int]:
    unknown = [c for c in prompt if c not in stoi]
    if unknown:
        # char models only know training vocab — drop or error clearly
        bad = "".join(sorted(set(unknown)))
        raise ValueError(
            f"prompt has characters not in training vocab: {bad!r}\n"
            f"Use only characters the model saw (ASCII Shakespeare-ish for that dataset)."
        )
    return [stoi[c] for c in prompt]


def decode(ids: list[int] | torch.Tensor, itos: dict[int, str]) -> str:
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return "".join(itos[i] for i in ids)


@torch.no_grad()
def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> torch.Tensor:
    """Sample tokens one by one (same idea as TinyLM.generate, with top-k)."""
    max_seq = model.config.max_seq_len  # type: ignore[attr-defined]
    for _ in range(max_new_tokens):
        window = input_ids[:, -max_seq:]
        logits, _ = model(window)

        logits = logits[:, -1, :]  # last position only

        if temperature <= 0:
            # greedy
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # mask everything below the k-th highest
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_id], dim=1)
    return input_ids


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        return torch.device("cpu")
    return torch.device(name)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate text with a trained TinyLM")
    p.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="checkpoints/shakespeare_dense.pt",
        help="path from python -m smollms.train --save ...",
    )
    p.add_argument("-p", "--prompt", type=str, default="To be")
    p.add_argument("-n", "--max-new-tokens", type=int, default=300)
    p.add_argument("--temp", "--temperature", dest="temperature", type=float, default=0.8)
    p.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="sample from top-k logits (0 = disabled)",
    )
    p.add_argument(
        "--greedy",
        action="store_true",
        help="always pick argmax (ignores temp/top-k)",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    model, stoi, itos = load_checkpoint(args.checkpoint, device)

    print(f"loaded {args.checkpoint}")
    print(
        f"  params={model.param_count_unique():,}  "
        f"vocab={model.config.vocab_size}  "
        f"d_model={model.config.d_model}  "
        f"layers={model.config.n_layers}  "
        f"max_seq={model.config.max_seq_len}"
    )
    print(f"prompt: {args.prompt!r}")
    print("---")

    ids = encode(args.prompt, stoi)
    if not ids:
        raise ValueError("prompt encoded to empty sequence")
    x = torch.tensor([ids], dtype=torch.long, device=device)

    temp = 0.0 if args.greedy else args.temperature
    top_k = None if args.greedy or args.top_k == 0 else args.top_k
    out = generate(
        model,
        x,
        max_new_tokens=args.max_new_tokens,
        temperature=temp,
        top_k=top_k,
    )
    text = decode(out[0], itos)
    print(text)
    print("---")


if __name__ == "__main__":
    main()
