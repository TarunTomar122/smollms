#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
exec env PYTHONPATH=src python3 -m smollms.train \
  --arch glm --data data/tinyshakespeare.txt --steps 2000 \
  --glm-dense-layers 1 --sparse-top-k 8 --seed 1337 --run-name glm-dense-prefix-sigmoid-moe
