"""Experiment run logger + compare helpers."""

import json
from pathlib import Path

import torch

from smollms.experiment.compare import plot_comparison, print_table, write_comparison_json
from smollms.experiment.run_logger import RunLogger, load_run, text_fingerprint
from smollms.model import TinyLM, TinyLMConfig


def test_text_fingerprint_stable():
    assert text_fingerprint("hello") == text_fingerprint("hello")
    assert text_fingerprint("hello") != text_fingerprint("world")


def test_run_logger_writes_bundle(tmp_path: Path):
    logger = RunLogger(tmp_path / "runs", arch="dense", run_name="unit")
    logger.log_train_args({"steps": 10, "lr": 1e-3})
    cfg = TinyLMConfig(vocab_size=10, d_model=16, n_layers=1, n_heads=2, max_seq_len=8)
    model = TinyLM(cfg)
    logger.log_model_config(cfg, model.param_count(), model.param_count_unique())
    logger.log_data_info(
        data_path="toy.txt",
        n_chars=100,
        vocab_size=10,
        train_windows=50,
        val_windows=10,
        block_size=8,
        text="abc" * 10,
    )
    logger.log_train(0, 2.5)
    logger.log_train(5, 2.0)
    logger.log_val(5, 2.1)
    logger.log_sample(5, "ab", "abcd", temperature=0.8, max_new_tokens=2)

    stoi = {chr(ord("a") + i): i for i in range(10)}
    itos = {i: c for c, i in stoi.items()}
    ckpt = logger.save_checkpoint(
        model=model, arch="dense", config=cfg, stoi=stoi, itos=itos
    )
    root = logger.finalize(final_train_loss=2.0, final_val_loss=2.1)

    assert ckpt.is_file()
    assert (root / "meta.json").is_file()
    assert (root / "train_args.json").is_file()
    assert (root / "model_config.json").is_file()
    assert (root / "data_info.json").is_file()
    assert (root / "history.json").is_file()
    assert (root / "samples.json").is_file()
    assert (root / "summary.json").is_file()
    assert (root / "metrics.jsonl").is_file()
    # plot may exist if matplotlib available
    loaded = load_run(root)
    assert loaded["summary"]["final_val_loss"] == 2.1
    assert len(loaded["history"]["train"]) == 2
    assert len(loaded["samples"]) == 1
    assert loaded["checkpoint"] is not None


def test_compare_two_runs(tmp_path: Path):
    runs = []
    for i, arch in enumerate(["dense", "kimi"]):
        logger = RunLogger(tmp_path / "runs", arch=arch, run_id=f"run_{arch}")
        logger.log_train(0, 3.0 - i * 0.1)
        logger.log_train(10, 2.0 - i * 0.1)
        logger.log_val(10, 2.2 - i * 0.1)
        logger.finalize(final_train_loss=2.0 - i * 0.1, final_val_loss=2.2 - i * 0.1)
        runs.append(load_run(logger.root))

    print_table(runs)
    out = tmp_path / "cmp"
    write_comparison_json(runs, out / "comparison.json")
    assert (out / "comparison.json").is_file()
    plot_comparison(runs, out / "loss_comparison.png")
    data = json.loads((out / "comparison.json").read_text())
    assert len(data) == 2
