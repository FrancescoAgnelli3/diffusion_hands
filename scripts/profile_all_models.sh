#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${DIFFUSION_HANDS_PYTHON:-$ROOT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

DIFFUSION_HANDS_PYTHON="$PYTHON_BIN" \
  "$PYTHON_BIN" "$ROOT_DIR/tools/profile_all_models.py" "$@"
