"""Minimal web UI to run the four smollms architectures.

Serves a bare HTML page and exposes POST /api/generate which loads
(a cached) checkpoint and generates text.

Usage (from repo root, venv on):

    python webapp/server.py [--port 8000]
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch

from smollms.infer import decode, encode, generate, load_checkpoint, resolve_device

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "webapp" / "index.html"

ARCHES: dict[str, dict[str, str]] = {
    "dense": {
        "name": "Dense / Qwen-style",
        "checkpoint": "runs/20260730_091426_dense_story-dense/checkpoint.pt",
    },
    "kimi": {
        "name": "Kimi-style (K2)",
        "checkpoint": "runs/20260730_091818_kimi2_story-kimi-full/checkpoint.pt",
    },
    "glm": {
        "name": "GLM-style",
        "checkpoint": "runs/20260730_094104_glm_story-glm-dsa/checkpoint.pt",
    },
    "deepseekv4": {
        "name": "DeepSeek V4-style",
        "checkpoint": "runs/20260730_095131_deepseekv4_story-v4-dense/checkpoint.pt",
    },
}

_models: dict[str, tuple[torch.nn.Module, dict[str, int], dict[int, str]]] = {}
_lock = threading.Lock()


def get_model(arch: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, int], dict[int, str]]:
    with _lock:
        if arch not in _models:
            ckpt = ROOT / ARCHES[arch]["checkpoint"]
            _models[arch] = load_checkpoint(ckpt, device)
    return _models[arch]


def handle_generate(body: dict, device: torch.device) -> dict:
    arch = body.get("arch")
    if arch not in ARCHES:
        return {"error": f"unknown arch {arch!r}; expected one of {sorted(ARCHES)}"}

    prompt = str(body.get("prompt", "To be")).strip()
    if not prompt:
        return {"error": "prompt is empty"}
    max_new = int(body.get("max_new_tokens", 200))
    temperature = float(body.get("temperature", 0.8))
    top_k = int(body.get("top_k", 40))

    model, stoi, itos = get_model(arch, device)
    try:
        ids = encode(prompt, stoi)
    except ValueError as e:
        return {"error": str(e)}

    if not ids:
        return {"error": "prompt encoded to empty sequence"}

    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = generate(
            model,
            x,
            max_new_tokens=max_new,
            temperature=temperature,
            top_k=top_k,
        )
    text = decode(out[0], itos)

    info = {
        "params": model.param_count_unique(),
        "vocab": model.config.vocab_size,
        "d_model": model.config.d_model,
        "layers": model.config.n_layers,
        "max_seq": model.config.max_seq_len,
    }
    return {"arch": arch, "prompt": prompt, "text": text, "model": info}


def make_handler(device: torch.device) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path != "/api/generate":
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                result = handle_generate(body, device)
                status = 400 if "error" in result else 200
            except Exception as e:  # noqa: BLE001
                result = {"error": f"{type(e).__name__}: {e}"}
                status = 500
            self._send(status, json.dumps(result).encode(), "application/json")

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = resolve_device(args.device)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(device))
    print(f"smollms web UI on http://127.0.0.1:{args.port}  (device={device})")
    print("checkpoints are loaded lazily on first run per architecture")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
