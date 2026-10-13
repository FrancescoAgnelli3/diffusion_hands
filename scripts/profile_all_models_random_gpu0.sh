#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${DIFFUSION_HANDS_PYTHON:-"$ROOT_DIR/.venv/bin/python"}

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
  "$ROOT_DIR/tools/profile_all_models_random.py" \
  --device cuda:0 \
  "$@"
