#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
exec env PYTHONPATH=src python3 -m smollms.train \
  --arch deepseekv4 --data data/tinyshakespeare.txt --steps 2000 \
  --v4-local-window 16 --v4-compression-ratio 4 \
  --v4-moe mixed --v4-hash-moe-layers 1 \
  --seed 1337 --run-name deepseek-v4-local-compressed-moe
