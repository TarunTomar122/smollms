"""Compare two (or more) training runs side by side.

    python -m smollms.compare runs/run_a runs/run_b
    python -m smollms.compare runs/run_a runs/run_b --out runs/comparisons/a_vs_b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from smollms.experiment.run_logger import load_run


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def print_table(runs: list[dict[str, Any]]) -> None:
    rows: list[tuple[str, list[str]]] = []

    def col(get) -> list[str]:
        return [_fmt(get(r)) for r in runs]

    rows.append(("run_id", col(lambda r: r.get("run_id"))))
    rows.append(("arch", col(lambda r: (r.get("meta") or {}).get("arch") or (r.get("summary") or {}).get("arch"))))
    rows.append(("params", col(lambda r: (r.get("model_config") or {}).get("param_count_unique"))))
    rows.append(("steps (train logs)", col(lambda r: len((r.get("history") or {}).get("train") or []))))
    rows.append(("final train loss", col(lambda r: (r.get("summary") or {}).get("final_train_loss"))))
    rows.append(("final val loss", col(lambda r: (r.get("summary") or {}).get("final_val_loss"))))
    rows.append(("best val loss", col(lambda r: (r.get("summary") or {}).get("best_val_loss"))))
    rows.append(("wall time (s)", col(lambda r: (r.get("summary") or {}).get("wall_time_sec"))))
    rows.append(("data fingerprint", col(lambda r: (r.get("data_info") or {}).get("text_sha256_16"))))
    rows.append(("vocab", col(lambda r: (r.get("data_info") or {}).get("vocab_size"))))

    # width
    label_w = max(len(k) for k, _ in rows)
    col_ws = []
    for i in range(len(runs)):
        col_ws.append(max(len(rows[j][1][i]) for j in range(len(rows))))
        col_ws[i] = max(col_ws[i], 8)

    header = " " * label_w + "  " + "  ".join(f"{'run'+str(i+1):>{w}}" for i, w in enumerate(col_ws))
    print(header)
    print("-" * len(header))
    for label, vals in rows:
        print(f"{label:<{label_w}}  " + "  ".join(f"{v:>{w}}" for v, w in zip(vals, col_ws)))


def plot_comparison(runs: list[dict[str, Any]], out_path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skip comparison plot")
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    for r in runs:
        hist = r.get("history") or {}
        train = hist.get("train") or []
        val = hist.get("val") or []
        arch = (r.get("meta") or {}).get("arch", "")
        run_name = (r.get("train_args") or {}).get("run_name") or r.get("run_id", "?")
        name = f"{arch}:{run_name}" if arch else str(run_name)
        if train:
            ax.plot(
                [e["step"] for e in train],
                [e["loss"] for e in train],
                label=f"{name} train",
                alpha=0.85,
            )
        if val:
            ax.plot(
                [e["step"] for e in val],
                [e["loss"] for e in val],
                label=f"{name} val",
                linestyle="--",
                marker="o",
                markersize=3,
            )
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("run comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def write_comparison_json(runs: list[dict[str, Any]], out_path: Path) -> None:
    payload = []
    for r in runs:
        payload.append(
            {
                "run_id": r.get("run_id"),
                "root": str(r.get("root")),
                "meta": r.get("meta"),
                "summary": r.get("summary"),
                "model_config": r.get("model_config"),
                "data_info": r.get("data_info"),
                "train_args": r.get("train_args"),
                "n_samples": len(r.get("samples") or []),
                "last_sample": (r.get("samples") or [None])[-1],
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare smollms training runs")
    p.add_argument("runs", nargs="+", help="paths to run directories under runs/")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="output directory for comparison plot + json (default: runs/comparisons/<id>)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    runs = [load_run(p) for p in args.runs]
    print_table(runs)

    if args.out:
        out_dir = Path(args.out)
    else:
        ids = "_vs_".join(r["run_id"][:24] for r in runs)
        out_dir = Path("runs") / "comparisons" / ids

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_comparison(runs, out_dir / "loss_comparison.png")
    write_comparison_json(runs, out_dir / "comparison.json")
    print(f"\nwrote comparison → {out_dir}")
    if plot_path:
        print(f"  plot: {plot_path}")

    # show last samples if any
    for r in runs:
        samples = r.get("samples") or []
        if not samples:
            continue
        last = samples[-1]
        print(f"\n--- sample [{r['run_id']}] step={last.get('step')} ---")
        print(last.get("text", "")[:500])


if __name__ == "__main__":
    main()
