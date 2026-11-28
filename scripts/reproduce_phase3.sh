#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
py=.venv/bin/python
"$py" scripts/assets.py
"$py" -m bench.phase3_dev
cmp configs/meetai.json .work/phase3/selected.json
for system in baseline meetai; do "$py" -m bench.phase3_run "$system"; done
for system in baseline meetai; do "$py" -m bench.phase3_scale "$system"; done
"$py" -m bench.phase3_probe
"$py" -m bench.phase3_score --packet
"$py" -m bench.phase3_score --dir .work/phase3 --out .work/phase3/scores.json
"$py" -m bench.phase3_verify --dir .work/phase3
