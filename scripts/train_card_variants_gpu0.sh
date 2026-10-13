#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_PATH=${1:-"$ROOT_DIR/configs/train_card_variants_vs_comusion.yaml"}
PYTHON_BIN=${DIFFUSION_HANDS_PYTHON:-"$ROOT_DIR/.venv/bin/python"}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found or not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Experiment config not found: $CONFIG_PATH" >&2
  exit 2
fi

START_MARKER=$(mktemp)
trap 'rm -f "$START_MARKER"' EXIT

echo "[train] Config: $CONFIG_PATH"
echo "[train] Physical GPU: 1"
cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES=1 \
DIFFUSION_HANDS_PYTHON="$PYTHON_BIN" \
  "$ROOT_DIR/scripts/run_all_models.sh" "$CONFIG_PATH"

latest_new_checkpoint() {
  local model_name=$1
  local model_root="$ROOT_DIR/out/diffusion_hands_runs/$model_name"
  if [[ ! -d "$model_root" ]]; then
    return 1
  fi
  find "$model_root" \
    -type f \
    -path '*/checkpoints/final.pt' \
    -newer "$START_MARKER" \
    -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

CARD_CHECKPOINT=$(latest_new_checkpoint card || true)
STAR_HAND_CHECKPOINT=$(latest_new_checkpoint star_hand || true)
FIHARD_CHECKPOINT=$(latest_new_checkpoint fihard || true)
COMUSION_CHECKPOINT=$(latest_new_checkpoint comusion || true)

missing=()
[[ -n "$CARD_CHECKPOINT" ]] || missing+=(card)
[[ -n "$STAR_HAND_CHECKPOINT" ]] || missing+=(star_hand)
[[ -n "$FIHARD_CHECKPOINT" ]] || missing+=(fihard)
[[ -n "$COMUSION_CHECKPOINT" ]] || missing+=(comusion)
if (( ${#missing[@]} > 0 )); then
  echo "[train] Missing new final checkpoints for: ${missing[*]}" >&2
  echo "[train] Inspect results/card_variants_vs_comusion_metrics.csv for failed runs." >&2
  exit 1
fi

echo
echo "[train] Final checkpoints:"
printf 'CARD_CHECKPOINT=%q\n' "$CARD_CHECKPOINT"
printf 'STAR_HAND_CHECKPOINT=%q\n' "$STAR_HAND_CHECKPOINT"
printf 'FIHARD_CHECKPOINT=%q\n' "$FIHARD_CHECKPOINT"
printf 'COMUSION_CHECKPOINT=%q\n' "$COMUSION_CHECKPOINT"
echo
echo "[train] Launch the paired Bayesian comparison on GPU 1 with:"
printf '%q ' "$ROOT_DIR/scripts/run_bayesian_signed_rank_gpu1.sh" \
  "$CARD_CHECKPOINT" \
  "$STAR_HAND_CHECKPOINT" \
  "$FIHARD_CHECKPOINT" \
  "$COMUSION_CHECKPOINT" \
  --experiment-config "$CONFIG_PATH" \
  --dataset assembly \
  --action-filter pick_up_screwd
echo

if [[ ${RUN_BAYESIAN_AFTER_TRAIN:-0} == 1 ]]; then
  BAYES_DATASET=${BAYES_DATASET:-assembly}
  BAYES_ACTION_FILTER=${BAYES_ACTION_FILTER:-pick_up_screwd}
  echo
  echo "[train] RUN_BAYESIAN_AFTER_TRAIN=1; starting paired comparison."
  "$ROOT_DIR/scripts/run_bayesian_signed_rank_gpu1.sh" \
    "$CARD_CHECKPOINT" \
    "$STAR_HAND_CHECKPOINT" \
    "$FIHARD_CHECKPOINT" \
    "$COMUSION_CHECKPOINT" \
    --experiment-config "$CONFIG_PATH" \
    --dataset "$BAYES_DATASET" \
    --action-filter "$BAYES_ACTION_FILTER"
fi
