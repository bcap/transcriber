#!/usr/bin/env bash
set -euo pipefail

cd $(dirname $0)

export LD_LIBRARY_PATH="$(readlink -f .)/.venv/lib/python3.14/site-packages/nvidia/cublas/lib"

uv run "./main.py" "$@"
