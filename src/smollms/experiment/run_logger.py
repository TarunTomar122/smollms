"""Experiment run logging — one folder per training job.

Layout
------
runs/<run_id>/
  meta.json          wall-clock, arch, device, param counts, status
  train_args.json    full CLI / hyperparameter dict
  model_config.json  architecture config (serializable)
  data_info.json     dataset path, sizes, vocab, optional text fingerprint
  metrics.jsonl      append-only events (train/val)
  history.json       same metrics as arrays (easy for plotting / compare)
  samples.json       generation snapshots during / after train
  summary.json       final numbers + best val
  plots/loss_curves.png
  checkpoint.pt      weights + arch + stoi/itos + run_id
  run.log            optional tee of important lines

Later: ``python -m smollms.compare run_a run_b`` loads two of these.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(obj: Any) -> Any:
    """Best-effort convert configs / paths / tensors to JSON."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if hasattr(obj, "__dict__") and not callable(obj):
        # e.g. argparse.Namespace
        return _jsonable(vars(obj))
    return str(obj)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_jsonable(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def text_fingerprint(text: str, n: int = 16) -> str:
    """Short hash so two runs can check they used the same corpus."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def make_run_id(arch: str, note: str | None = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_note = ""
    if note:
        safe_note = "_" + "".join(c if c.isalnum() or c in "-_" else "-" for c in note)[:40]
    return f"{stamp}_{arch}{safe_note}"


class RunLogger:
    """Creates ``runs/<run_id>/`` and records everything needed for later compare."""

    def __init__(
        self,
        runs_dir: str | Path,
        arch: str,
        run_name: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.runs_dir = Path(runs_dir)
        self.arch = arch
        self.run_id = run_id or make_run_id(arch, run_name)
        self.root = self.runs_dir / self.run_id
        self.plots_dir = self.root / "plots"
        self.root.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.t0 = time.time()
        self.train_history: list[dict[str, Any]] = []
        self.val_history: list[dict[str, Any]] = []
        self.samples: list[dict[str, Any]] = []
        self._metrics_path = self.root / "metrics.jsonl"
        # truncate metrics file for a fresh run
        self._metrics_path.write_text("", encoding="utf-8")

        _write_json(
            self.root / "meta.json",
            {
                "run_id": self.run_id,
                "arch": arch,
                "status": "running",
                "created_at": _utc_now(),
                "finished_at": None,
                "wall_time_sec": None,
            },
        )

    # ----- setup dumps -----

    def log_train_args(self, args: Any) -> None:
        _write_json(self.root / "train_args.json", args)

    def log_model_config(self, config: Any, param_count: int, param_count_unique: int) -> None:
        _write_json(
            self.root / "model_config.json",
            {
                "config": config,
                "param_count": param_count,
                "param_count_unique": param_count_unique,
                "config_type": type(config).__name__,
            },
        )

    def log_data_info(
        self,
        *,
        data_path: str | None,
        n_chars: int,
        vocab_size: int,
        train_windows: int,
        val_windows: int,
        block_size: int,
        text: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        info: dict[str, Any] = {
            "data_path": data_path,
            "n_chars": n_chars,
            "vocab_size": vocab_size,
            "train_windows": train_windows,
            "val_windows": val_windows,
            "block_size": block_size,
        }
        if text is not None:
            info["text_sha256_16"] = text_fingerprint(text)
            info["n_chars_check"] = len(text)
        if extra:
            info.update(extra)
        _write_json(self.root / "data_info.json", info)

    # ----- live metrics -----

    def _append_metrics(self, row: dict[str, Any]) -> None:
        row = {**row, "wall_time_sec": round(time.time() - self.t0, 4)}
        with self._metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        kind = row.get("type", "train")
        if kind == "val":
            self.val_history.append(row)
        else:
            self.train_history.append(row)

    def log_train(self, step: int, loss: float, **extra: Any) -> None:
        self._append_metrics({"type": "train", "step": step, "loss": float(loss), **extra})

    def log_val(self, step: int, loss: float, **extra: Any) -> None:
        self._append_metrics({"type": "val", "step": step, "loss": float(loss), **extra})

    def log_sample(
        self,
        step: int,
        prompt: str,
        text: str,
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        **extra: Any,
    ) -> None:
        row = {
            "step": step,
            "prompt": prompt,
            "text": text,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "wall_time_sec": round(time.time() - self.t0, 4),
            **extra,
        }
        self.samples.append(row)
        _write_json(self.root / "samples.json", self.samples)

    # ----- finish -----

    def save_history(self) -> None:
        _write_json(
            self.root / "history.json",
            {
                "train": self.train_history,
                "val": self.val_history,
            },
        )

    def save_plots(self) -> Path | None:
        """Write loss curves PNG. Returns path or None if matplotlib missing / no data."""
        if not self.train_history and not self.val_history:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — skip plots (pip install matplotlib)")
            return None

        fig, ax = plt.subplots(figsize=(8, 4.5))
        if self.train_history:
            ax.plot(
                [r["step"] for r in self.train_history],
                [r["loss"] for r in self.train_history],
                label="train loss",
                alpha=0.9,
            )
        if self.val_history:
            ax.plot(
                [r["step"] for r in self.val_history],
                [r["loss"] for r in self.val_history],
                label="val loss",
                marker="o",
                markersize=3,
            )
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(f"{self.run_id}  ({self.arch})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = self.plots_dir / "loss_curves.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out

    def save_checkpoint(
        self,
        *,
        model: torch.nn.Module,
        arch: str,
        config: Any,
        stoi: dict[str, int],
        itos: dict[int, str],
        filename: str = "checkpoint.pt",
        extra: dict[str, Any] | None = None,
    ) -> Path:
        path = self.root / filename
        payload: dict[str, Any] = {
            "arch": arch,
            "run_id": self.run_id,
            "model": model.state_dict(),
            "config": config,
            "stoi": stoi,
            "itos": itos,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return path

    def finalize(
        self,
        *,
        status: str = "completed",
        final_train_loss: float | None = None,
        final_val_loss: float | None = None,
        extra_summary: dict[str, Any] | None = None,
    ) -> Path:
        self.save_history()
        plot_path = self.save_plots()
        wall = time.time() - self.t0

        best_val = None
        if self.val_history:
            best_val = min(self.val_history, key=lambda r: r["loss"])

        summary: dict[str, Any] = {
            "run_id": self.run_id,
            "arch": self.arch,
            "status": status,
            "wall_time_sec": round(wall, 3),
            "n_train_logs": len(self.train_history),
            "n_val_logs": len(self.val_history),
            "n_samples": len(self.samples),
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "best_val_loss": None if best_val is None else best_val["loss"],
            "best_val_step": None if best_val is None else best_val["step"],
            "plot": None if plot_path is None else str(plot_path.relative_to(self.root)),
        }
        if extra_summary:
            summary.update(extra_summary)
        _write_json(self.root / "summary.json", summary)

        meta = {
            "run_id": self.run_id,
            "arch": self.arch,
            "status": status,
            "created_at": None,
            "finished_at": _utc_now(),
            "wall_time_sec": round(wall, 3),
        }
        # preserve created_at if present
        meta_path = self.root / "meta.json"
        if meta_path.is_file():
            try:
                old = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["created_at"] = old.get("created_at")
            except json.JSONDecodeError:
                pass
        if meta["created_at"] is None:
            meta["created_at"] = _utc_now()
        _write_json(meta_path, meta)
        return self.root


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load a finished (or in-progress) run folder into a plain dict."""
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"run dir not found: {root}")

    def _read(name: str) -> Any | None:
        p = root / name
        if not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    history = _read("history.json") or {"train": [], "val": []}
    # rebuild history from jsonl if history missing but metrics exist
    metrics_path = root / "metrics.jsonl"
    if not history.get("train") and metrics_path.is_file():
        train, val = [], []
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            (val if row.get("type") == "val" else train).append(row)
        history = {"train": train, "val": val}

    return {
        "root": root,
        "run_id": root.name,
        "meta": _read("meta.json"),
        "train_args": _read("train_args.json"),
        "model_config": _read("model_config.json"),
        "data_info": _read("data_info.json"),
        "history": history,
        "samples": _read("samples.json") or [],
        "summary": _read("summary.json"),
        "checkpoint": root / "checkpoint.pt" if (root / "checkpoint.pt").is_file() else None,
        "plots": {
            "loss_curves": root / "plots" / "loss_curves.png"
            if (root / "plots" / "loss_curves.png").is_file()
            else None
        },
    }
