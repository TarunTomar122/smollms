#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
exec env PYTHONPATH=src python3 -m smollms.train \
  --arch glm --glm-attention dense --data data/tinyshakespeare.txt --steps 2000 \
  --glm-dense-layers 1 --seed 1337 --run-name glm-dense-attention-sigmoid-moe
