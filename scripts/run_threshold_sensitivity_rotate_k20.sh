#!/usr/bin/env bash
# Suite-E2-compatible seed-0 sensitivity: matching soft GRIP, preprocessing,
# source-file cap, DLow checkpoint, threshold, and K.  It intentionally uses
# one seed; set the output root per seed if a multi-seed aggregation is wanted.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${DIFFUSION_HANDS_PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

OUTPUT_ROOT="${THRESHOLD_SENSITIVITY_OUTPUT_ROOT:-$ROOT_DIR/results/threshold_sensitivity_rotate_20261019/suite_e2_seed0}"
DATA_DIR="${ASSEMBLY_DATA_DIR:-/mnt/pve/Turing-Storage2/AssemblyHands/assembly101-download-scripts/data_our}"
CANDIDATE_BATCH_SIZE="${CIT_CANDIDATE_BATCH_SIZE:-5}"
MAX_SOURCE_FILES="${THRESHOLD_SENSITIVITY_MAX_SOURCE_FILES:-2000}"

cd "$ROOT_DIR"

"$PYTHON_BIN" tools/reevaluate_cit_checkpoint.py \
  --config vendor/splineeqnet/out/diffusion_hands_runs/cit_grip_phase_coupled_soft/assembly_rotate_20261014_225954_cit_grip_phase_coupled_soft_seed0/card_best_config.json \
  --checkpoint out/diffusion_hands_runs/cit_grip_phase_coupled_soft/assembly_rotate_20261014_225954_cit_grip_phase_coupled_soft_seed0/checkpoints/final.pt \
  --output "$OUTPUT_ROOT/grip_seed0_k20_metrics.json" \
  --prediction-cache "$OUTPUT_ROOT/caches/grip_seed0_k20.pt" \
  --data-dir "$DATA_DIR" \
  --action-filter rotate \
  --seed 0 \
  --max-source-files "$MAX_SOURCE_FILES" \
  --num-candidates 20 \
  --candidate-batch-size "$CANDIDATE_BATCH_SIZE" \
  --cache-only

"$PYTHON_BIN" tools/reevaluate_dlow_on_reference_cache.py \
  --config vendor/dlow/motion_pred/cfg/threshold_rotate_dlow_seed0.yml \
  --checkpoint out/diffusion_hands_runs/dlow_cvae/assembly_rotate_20261014_225954_dlow_cvae_seed0/checkpoints/final.pt \
  --reference-cache "$OUTPUT_ROOT/caches/grip_seed0_k20.pt" \
  --output-cache "$OUTPUT_ROOT/caches/dlow_on_grip_seed0_k20.pt" \
  --num-candidates 20 \
  --batch-size 128 \
  --seed 0

"$PYTHON_BIN" tools/sweep_multimodal_threshold.py \
  --dlow-cache "$OUTPUT_ROOT/caches/dlow_on_grip_seed0_k20.pt" \
  --grip-cache "$OUTPUT_ROOT/caches/grip_seed0_k20.pt" \
  --thresholds 0,1,2,3,4,5,5.9545321465,6.5499853611,7,8,10,12,16,20,30 \
  --output-dir "$OUTPUT_ROOT/seed0_k20"
