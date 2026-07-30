#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
exec env PYTHONPATH=src python3 -m smollms.train \
  --arch dense --data data/tinyshakespeare.txt --steps 2000 --seed 1337 --run-name dense-2000
