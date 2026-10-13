#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 CARD_CHECKPOINT STAR_HAND_CHECKPOINT FIHARD_CHECKPOINT COMUSION_CHECKPOINT [extra arguments...]" >&2
  exit 2
fi

CARD_CHECKPOINT=$1
STAR_HAND_CHECKPOINT=$2
FIHARD_CHECKPOINT=$3
COMUSION_CHECKPOINT=$4
shift 4

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${DIFFUSION_HANDS_PYTHON:-"$ROOT_DIR/.venv/bin/python"}

CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" "$ROOT_DIR/tools/bayesian_signed_rank_test.py" \
  --device cuda:0 \
  --card-checkpoint "$CARD_CHECKPOINT" \
  --star-hand-checkpoint "$STAR_HAND_CHECKPOINT" \
  --fihard-checkpoint "$FIHARD_CHECKPOINT" \
  --comusion-checkpoint "$COMUSION_CHECKPOINT" \
  "$@"
