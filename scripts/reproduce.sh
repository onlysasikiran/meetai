#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/assets.py
.venv/bin/python -m pytest -q
.venv/bin/python -m bench.tune
cmp configs/baseline.json .work/retune/baseline.json
.venv/bin/python -m bench.run
.venv/bin/python -m bench.score --run .work/replay/checkpoint.json --out .work/replay/scores.json
.venv/bin/python -m bench.verify --run .work/replay/checkpoint.json --out .work/replay/validation.json
