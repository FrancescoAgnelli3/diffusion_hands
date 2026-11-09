#!/usr/bin/env bash
# E1 (30 observed / 10 forecast) threshold sensitivity for Assembly ``remove``.
# Both systems are evaluated on one fixed GRIP cache, so only the multimodal
# threshold varies. Defaults point to the archived E1, seed-0 cohort.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${DIFFUSION_HANDS_PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

SUITE_ROOT="${E1_REMOVE_SUITE_ROOT:-$ROOT_DIR/results/forecasting_suite/recalibrated/E1}"
OUTPUT_ROOT="${THRESHOLD_SENSITIVITY_OUTPUT_ROOT:-$ROOT_DIR/results/threshold_sensitivity_remove_e1/suite_e1_seed0}"
DATA_DIR="${ASSEMBLY_DATA_DIR:-/mnt/pve/Turing-Storage2/AssemblyHands/assembly101-download-scripts/data_our}"
MAX_SOURCE_FILES="${THRESHOLD_SENSITIVITY_MAX_SOURCE_FILES:-2000}"
CANDIDATE_BATCH_SIZE="${CIT_CANDIDATE_BATCH_SIZE:-5}"
THRESHOLDS="${THRESHOLD_SENSITIVITY_THRESHOLDS:-0,1,2,3,3.262270927429199,4,5,6,8,10,12,16,20}"

CIT_RUN="${E1_REMOVE_CIT_RUN:-$SUITE_ROOT/runs/checkpoint_runs/cit_grip_phase_coupled_soft/assembly_remove_20261020_153135_cit_grip_phase_coupled_soft_seed0}"
DLOW_RUN="${E1_REMOVE_DLOW_RUN:-$SUITE_ROOT/runs/checkpoint_runs/dlow_cvae/assembly_remove_20261020_153135_dlow_cvae_seed0}"
CIT_CONFIG="${E1_REMOVE_CIT_CONFIG:-$ROOT_DIR/vendor/splineeqnet/out/diffusion_hands_runs/cit_grip_phase_coupled_soft/assembly_remove_20261020_153135_cit_grip_phase_coupled_soft_seed0/card_best_config.json}"
DLOW_CONFIG="${E1_REMOVE_DLOW_CONFIG:-$ROOT_DIR/configs/threshold_sensitivity/remove_e1_dlow_seed0.yaml}"
CIT_CHECKPOINT="${E1_REMOVE_CIT_CHECKPOINT:-$CIT_RUN/checkpoints/00_final.pt}"
DLOW_CHECKPOINT="${E1_REMOVE_DLOW_CHECKPOINT:-$DLOW_RUN/checkpoints/00_final.pt}"

for required in "$CIT_CONFIG" "$DLOW_CONFIG" "$CIT_CHECKPOINT" "$DLOW_CHECKPOINT"; do
  if [[ ! -f "$required" ]]; then
    echo "Required E1/remove artifact is missing: $required" >&2
    echo "Set the matching E1_REMOVE_* environment variable to override it." >&2
    exit 2
  fi
done

cd "$ROOT_DIR"

"$PYTHON_BIN" tools/reevaluate_cit_checkpoint.py \
  --config "$CIT_CONFIG" \
  --checkpoint "$CIT_CHECKPOINT" \
  --output "$OUTPUT_ROOT/grip_seed0_k20_metrics.json" \
  --prediction-cache "$OUTPUT_ROOT/caches/grip_seed0_k20.pt" \
  --data-dir "$DATA_DIR" \
  --action-filter remove \
  --seed 0 \
  --max-source-files "$MAX_SOURCE_FILES" \
  --num-candidates 20 \
  --candidate-batch-size "$CANDIDATE_BATCH_SIZE" \
  --cache-only

"$PYTHON_BIN" tools/reevaluate_dlow_on_reference_cache.py \
  --config "$DLOW_CONFIG" \
  --checkpoint "$DLOW_CHECKPOINT" \
  --reference-cache "$OUTPUT_ROOT/caches/grip_seed0_k20.pt" \
  --output-cache "$OUTPUT_ROOT/caches/dlow_on_grip_seed0_k20.pt" \
  --num-candidates 20 \
  --batch-size 128 \
  --seed 0

"$PYTHON_BIN" tools/sweep_multimodal_threshold.py \
  --dlow-cache "$OUTPUT_ROOT/caches/dlow_on_grip_seed0_k20.pt" \
  --grip-cache "$OUTPUT_ROOT/caches/grip_seed0_k20.pt" \
  --thresholds "$THRESHOLDS" \
  --output-dir "$OUTPUT_ROOT/seed0_k20"
