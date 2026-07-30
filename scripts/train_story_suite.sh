#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

seed="${SEED:-1337}"
device="${DEVICE:-cpu}"
steps="${STEPS:-2000}"
common="--data data/tinyshakespeare.txt --steps $steps --device $device --seed $seed"

run() {
  PYTHONPATH=src python3 -m smollms.train "$@" $common
}

run --arch dense --run-name story-dense
run --arch kimi2 --hybrid-pattern all_full --attn-res false --run-name story-kimi-full
run --arch kimi2 --hybrid-pattern 1L1F --attn-res false --run-name story-kimi-1l1f
run --arch glm --glm-attention sparse --glm-dense-layers 1 --run-name story-glm-dsa
run --arch glm --glm-attention dense --glm-dense-layers 1 --run-name story-glm-dense-attn
run --arch deepseekv4 --v4-moe dense --run-name story-v4-dense
run --arch deepseekv4 --v4-moe mixed --v4-hash-moe-layers 1 --run-name story-v4-moe
