#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${MEETAI_PYTHON:-python3.12}"
"$python_bin" -m venv .venv
# A native build selects the instruction set of the actual CPU and its C library.
CC="${MEETAI_CC:-gcc}" CXX="${MEETAI_CXX:-g++}" CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}" \
CMAKE_ARGS='-DGGML_NATIVE=ON -DGGML_OPENMP=ON -DLLAMA_CURL=OFF' \
    .venv/bin/python -m pip install --no-binary=llama-cpp-python -r requirements.lock
.venv/bin/python scripts/assets.py
