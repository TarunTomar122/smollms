"""Minimal training loop for character-level LMs (dense / kimi / …).

Every run writes a full experiment folder under ``runs/`` (metrics, plots,
samples, config, checkpoint) so you can compare arches later.

Tiny Shakespeare::

    python -m smollms.train --arch dense --data data/tinyshakespeare.txt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from smollms.data.char_data import load_text, prepare_char_data
from smollms.experiment.run_logger import RunLogger
from smollms.model import TinyLM, TinyLMConfig
from smollms.variants.deepseek_v4 import V4ToyLM, V4ToyLMConfig
from smollms.variants.glm import GLMLM, GLMLMConfig
from smollms.variants.kimi import KimiLM, KimiLMConfig


def estimate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total += loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


def _build_model(args: argparse.Namespace, vocab_size: int):
    arch = args.arch
    if arch == "dense":
        config = TinyLMConfig(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            max_seq_len=args.block_size,
            qk_norm=True,
            dropout=args.dropout,
            tie_weights=True,
        )
        return arch, config, TinyLM(config)
    if arch in ("kimi", "kimi2"):
        step = getattr(args, "kimi_step", 2 if arch == "kimi2" else 1)
        if arch == "kimi2":
            step = max(step, 2)
        # Step 2 defaults to a KDA/MLA hybrid, but an explicit all_full is a
        # valid ablation and must remain full attention.
        if step >= 2:
            hybrid = args.hybrid_pattern or "3L1F"
            use_attn_res = True if args.attn_res is None else args.attn_res
        else:
            hybrid = args.hybrid_pattern or "all_full"
            use_attn_res = False if args.attn_res is None else args.attn_res
        config = KimiLMConfig(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            d_latent=args.d_latent,
            n_experts=args.n_experts,
            n_active=args.n_active,
            n_shared=args.n_shared,
            max_seq_len=args.block_size,
            qk_norm=True,
            dropout=args.dropout,
            aux_loss_weight=args.aux_loss_weight,
            tie_weights=True,
            kimi_step=step,
            hybrid_pattern=hybrid,
            use_attn_res=use_attn_res,
            attn_res_depth=args.attn_res_depth,
        )
        log_arch = "kimi2" if step >= 2 else "kimi"
        return log_arch, config, KimiLM(config)
    if arch == "glm":
        config = GLMLMConfig(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            attention_kind=args.glm_attention,
            sparse_top_k=args.sparse_top_k,
            n_dense_layers=args.glm_dense_layers,
            n_experts=args.n_experts,
            n_active=args.n_active,
            n_shared=args.n_shared,
            max_seq_len=args.block_size,
            qk_norm=True,
            dropout=args.dropout,
            tie_weights=True,
        )
        return arch, config, GLMLM(config)
    if arch == "deepseekv4":
        config = V4ToyLMConfig(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            local_window=args.v4_local_window,
            compression_ratio=args.v4_compression_ratio,
            moe_mode=args.v4_moe,
            n_hash_moe_layers=args.v4_hash_moe_layers,
            n_experts=args.n_experts,
            n_active=args.n_active,
            n_shared=args.n_shared,
            max_seq_len=args.block_size,
            qk_norm=True,
            dropout=args.dropout,
            tie_weights=True,
        )
        return arch, config, V4ToyLM(config)
    raise SystemExit(f"unknown --arch {arch!r}")


def _sample_text(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    if ids.numel() == 0:
        ids = torch.zeros(1, 1, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature)
    return tokenizer.decode(out[0])


def _prepare_data(args: argparse.Namespace):
    """Return tokenizer, train_ds, val_ds, and Tiny Shakespeare metadata."""
    text = load_text(args.data)
    tokenizer, train_ds, val_ds = prepare_char_data(text, block_size=args.block_size)
    info = {
        "dataset": "shakespeare",
        "data_path": args.data,
        "n_chars": len(text),
        "vocab_size": tokenizer.vocab_size,
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "block_size": args.block_size,
        "text": text,
    }
    print(
        f"dataset=shakespeare  vocab={tokenizer.vocab_size}  train_windows={len(train_ds)}  "
        f"val_windows={len(val_ds)}  block_size={args.block_size}"
    )
    return tokenizer, train_ds, val_ds, info


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        device = torch.device("cpu")

    # Resolve arch label early for run id (kimi2 vs kimi)
    pre_arch = args.arch
    if pre_arch == "kimi2" or (
        pre_arch == "kimi" and getattr(args, "kimi_step", 1) >= 2
    ):
        run_arch = "kimi2"
    else:
        run_arch = pre_arch

    logger = RunLogger(
        runs_dir=args.runs_dir,
        arch=run_arch,
        run_name=args.run_name,
    )
    logger.log_train_args(args)
    print(f"run_id={logger.run_id}")
    print(f"run_dir={logger.root}")

    tokenizer, train_ds, val_ds, data_info = _prepare_data(args)
    logger.log_data_info(
        data_path=data_info.get("data_path"),
        n_chars=data_info.get("n_chars", data_info.get("n_train", 0)),
        vocab_size=data_info["vocab_size"],
        train_windows=data_info["train_windows"],
        val_windows=data_info["val_windows"],
        block_size=data_info["block_size"],
        text=data_info.get("text"),
        extra={k: v for k, v in data_info.items() if k not in ("text",)},
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    arch, config, model = _build_model(args, tokenizer.vocab_size)
    model = model.to(device)
    n_params = model.param_count()
    n_unique = model.param_count_unique()
    logger.log_model_config(config, param_count=n_params, param_count_unique=n_unique)
    print(f"arch={arch}  params (unique)={n_unique:,}  device={device}")
    print(f"config={config}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    model.train()
    step = 0
    t0 = time.time()
    data_iter = iter(train_loader)
    last_train_loss = None
    def _run_eval(step_i: int) -> dict[str, float]:
        val_loss = estimate_loss(model, val_loader, device)
        print(f"  → val_loss={val_loss:.4f}")
        logger.log_val(step_i, val_loss)
        return {"val_loss": val_loss}

    while step < args.steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        last_train_loss = loss.item()

        if step % args.log_every == 0 or step == args.steps - 1:
            dt = time.time() - t0
            print(f"step {step:5d}/{args.steps}  loss={last_train_loss:.4f}  ({dt:.1f}s)")
            logger.log_train(step, last_train_loss)

        if step > 0 and step % args.eval_every == 0:
            last_metrics = _run_eval(step)

        if args.sample_every > 0 and step > 0 and step % args.sample_every == 0:
            sample = _sample_text(
                model,
                tokenizer,
                device,
                args.sample_prompt,
                args.sample_tokens,
                args.temperature,
            )
            print(f"  → sample@step{step}: {sample[:120]!r}...")
            logger.log_sample(
                step,
                args.sample_prompt,
                sample,
                temperature=args.temperature,
                max_new_tokens=args.sample_tokens,
            )
            model.train()

        step += 1

    last_metrics = _run_eval(args.steps - 1)
    final_val = float(last_metrics.get("val_loss", 0.0))
    print(f"done. final val_loss={final_val:.4f}")

    final_sample = _sample_text(
        model,
        tokenizer,
        device,
        args.sample_prompt,
        args.sample_tokens,
        args.temperature,
    )
    print("--- sample ---")
    print(final_sample)
    print("--------------")
    logger.log_sample(
        args.steps - 1,
        args.sample_prompt,
        final_sample,
        temperature=args.temperature,
        max_new_tokens=args.sample_tokens,
        final=True,
    )

    ckpt_path = logger.save_checkpoint(
        model=model,
        arch=arch,
        config=config,
        stoi=tokenizer.stoi,
        itos=tokenizer.itos,
        extra={"dataset": "shakespeare", "data_info": {k: v for k, v in data_info.items() if k != "text"}},
    )
    print(f"checkpoint → {ckpt_path}")

    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "arch": arch,
                "run_id": logger.run_id,
                "model": model.state_dict(),
                "config": config,
                "stoi": tokenizer.stoi,
                "itos": tokenizer.itos,
                "dataset": "shakespeare",
            },
            path,
        )
        print(f"also saved → {path}")

    extra_summary = {
        "device": str(device),
        "param_count_unique": n_unique,
        "checkpoint": "checkpoint.pt",
        "dataset": "shakespeare",
    }

    run_root = logger.finalize(
        status="completed",
        final_train_loss=last_train_loss,
        final_val_loss=final_val,
        extra_summary=extra_summary,
    )
    print(f"run complete → {run_root}")
    print(f"  summary: {run_root / 'summary.json'}")
    print(f"  plots:   {run_root / 'plots' / 'loss_curves.png'}")
    return run_root


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a smollms LM (logs full experiment run)")
    p.add_argument(
        "--arch",
        type=str,
        default="dense",
        choices=["dense", "kimi", "kimi2", "glm", "deepseekv4"],
        help="dense | kimi | kimi2 | glm | deepseekv4 (local + compressed memory)",
    )
    p.add_argument("--data", type=str, default=None, help="Tiny Shakespeare text file")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-kv-heads", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=1337, help="PyTorch seed recorded with the run")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--save",
        type=str,
        default=None,
        help="optional extra copy of checkpoint.pt at this path",
    )
    # experiment logging
    p.add_argument("--runs-dir", type=str, default="runs", help="root folder for experiment runs")
    p.add_argument("--run-name", type=str, default=None, help="optional note baked into run_id")
    p.add_argument("--sample-prompt", type=str, default="To be")
    p.add_argument("--sample-tokens", type=int, default=120)
    p.add_argument(
        "--sample-every",
        type=int,
        default=0,
        help="log a generation every N steps (0 = only at end)",
    )
    # kimi / kimi2 knobs (ignored for dense)
    p.add_argument("--d-latent", type=int, default=None)
    p.add_argument("--n-experts", type=int, default=4)
    p.add_argument("--n-active", type=int, default=2)
    p.add_argument("--n-shared", type=int, default=1)
    p.add_argument("--aux-loss-weight", type=float, default=0.01)
    p.add_argument(
        "--kimi-step",
        type=int,
        default=1,
        help="1=LatentMoE only; 2=+hybrid KDA/MLA + AttnRes (also via --arch kimi2)",
    )
    p.add_argument(
        "--hybrid-pattern",
        type=str,
        default=None,
        choices=["all_full", "3L1F", "1L1F"],
        help="Kimi token-mixer pattern (kimi2 defaults to 3L1F; kimi step 1 to all_full)",
    )
    p.add_argument(
        "--attn-res",
        type=lambda s: s.lower() in ("1", "true", "yes", "y"),
        default=None,
        help="enable AttnRes (default on for kimi2, off for kimi step1)",
    )
    p.add_argument("--attn-res-depth", type=int, default=2)
    p.add_argument(
        "--sparse-top-k",
        type=int,
        default=8,
        help="number of causal tokens each DSA head keeps (glm sparse attention only)",
    )
    p.add_argument(
        "--glm-attention",
        choices=["sparse", "dense"],
        default="sparse",
        help="GLM token mixer; dense is the controlled DSA ablation",
    )
    p.add_argument(
        "--glm-dense-layers",
        type=int,
        default=1,
        help="dense SwiGLU prefix before routed MoE layers (glm only)",
    )
    p.add_argument("--v4-local-window", type=int, default=16, help="raw local window (deepseekv4 only)")
    p.add_argument("--v4-compression-ratio", type=int, default=4, help="tokens per learned compressed-memory entry (deepseekv4 only)")
    p.add_argument("--v4-moe", choices=["dense", "mixed"], default="dense", help="dense FFN or hash-then-routed MoE (deepseekv4 only)")
    p.add_argument("--v4-hash-moe-layers", type=int, default=1, help="static hash-MoE prefix length (deepseekv4 mixed only)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.data is None:
        # prefer downloaded file if present
        cand = Path("data/tinyshakespeare.txt")
        args.data = str(cand) if cand.is_file() else None
    train(args)


if __name__ == "__main__":
    main()
