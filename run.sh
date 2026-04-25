#!/usr/bin/env bash
set -euo pipefail

cd $(dirname $0)

CUDA_LIB_PATH="$(ls -d ./.venv/lib/python*/site-packages/nvidia/cublas/lib | head -n 1)"

export LD_LIBRARY_PATH="${CUDA_LIB_PATH}:${LD_LIBRARY_PATH:-}"

uv run "./main.py" "$@"
