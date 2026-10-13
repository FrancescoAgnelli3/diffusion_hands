#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_PATH=${PROFILE_CONFIG:-"$ROOT_DIR/configs/experiment.yaml"}
PYTHON_BIN=${DIFFUSION_HANDS_PYTHON:-"$ROOT_DIR/.venv/bin/python"}

GPU_INDEX=1
POLL_SECONDS=${GPU_POLL_SECONDS:-60}
REQUIRED_IDLE_CHECKS=${GPU_REQUIRED_IDLE_CHECKS:-2}
MAX_UTILIZATION=${GPU_MAX_UTILIZATION:-5}
MAX_MEMORY_MIB=${GPU_MAX_MEMORY_MIB:-512}

for setting in POLL_SECONDS REQUIRED_IDLE_CHECKS MAX_UTILIZATION MAX_MEMORY_MIB; do
  value=${!setting}
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "[profile] $setting must be a non-negative integer, got: $value" >&2
    exit 2
  fi
done
if (( REQUIRED_IDLE_CHECKS < 1 )); then
  echo "[profile] GPU_REQUIRED_IDLE_CHECKS must be at least 1." >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[profile] nvidia-smi is unavailable; cannot monitor GPU $GPU_INDEX." >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[profile] Python environment not found or not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[profile] Experiment config not found: $CONFIG_PATH" >&2
  exit 2
fi
if ! nvidia-smi --id="$GPU_INDEX" --query-gpu=index --format=csv,noheader,nounits \
    >/dev/null 2>&1; then
  echo "[profile] Physical GPU $GPU_INDEX does not exist or is inaccessible." >&2
  exit 2
fi
GPU_UUID=$(
  nvidia-smi --id="$GPU_INDEX" \
    --query-gpu=uuid \
    --format=csv,noheader,nounits \
    | awk 'NR == 1 {gsub(/[[:space:]]/, "", $0); print; exit}'
)
if [[ -z "$GPU_UUID" ]]; then
  echo "[profile] Could not resolve the UUID of physical GPU $GPU_INDEX." >&2
  exit 2
fi

gpu_has_compute_processes() {
  local process_pids
  process_pids=$(
    nvidia-smi \
      --query-compute-apps=gpu_uuid,pid \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk -F',' -v target_uuid="$GPU_UUID" '{
          gsub(/[[:space:]]/, "", $1)
          gsub(/[[:space:]]/, "", $2)
          if ($1 == target_uuid && $2 != "") print $2
        }'
  )
  [[ -n "$process_pids" ]]
}

read_gpu_state() {
  nvidia-smi --id="$GPU_INDEX" \
    --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits \
    | awk -F',' 'NR == 1 {
        gsub(/[[:space:]]/, "", $1)
        gsub(/[[:space:]]/, "", $2)
        print $1, $2
      }'
}

echo "[profile] Waiting for physical GPU $GPU_INDEX to become idle."
echo "[profile] Idle criteria: no compute processes, utilization <= ${MAX_UTILIZATION}%, memory <= ${MAX_MEMORY_MIB} MiB."
echo "[profile] Poll interval: ${POLL_SECONDS}s; consecutive idle checks required: $REQUIRED_IDLE_CHECKS."

idle_checks=0
while (( idle_checks < REQUIRED_IDLE_CHECKS )); do
  read -r utilization memory_mib < <(read_gpu_state)

  if ! gpu_has_compute_processes \
      && (( utilization <= MAX_UTILIZATION )) \
      && (( memory_mib <= MAX_MEMORY_MIB )); then
    ((idle_checks += 1))
    echo "[profile] GPU $GPU_INDEX idle check ${idle_checks}/${REQUIRED_IDLE_CHECKS}: utilization=${utilization}%, memory=${memory_mib} MiB."
  else
    idle_checks=0
    echo "[profile] GPU $GPU_INDEX busy: utilization=${utilization}%, memory=${memory_mib} MiB; checking again in ${POLL_SECONDS}s."
  fi

  if (( idle_checks < REQUIRED_IDLE_CHECKS )); then
    sleep "$POLL_SECONDS"
  fi
done

# Recheck immediately before launch to reduce the race between the last poll and
# starting the profiler. There is no cross-user GPU reservation primitive in
# nvidia-smi, so a scheduler should be preferred on managed clusters.
read -r utilization memory_mib < <(read_gpu_state)
if gpu_has_compute_processes \
    || (( utilization > MAX_UTILIZATION )) \
    || (( memory_mib > MAX_MEMORY_MIB )); then
  echo "[profile] GPU $GPU_INDEX became busy before launch; restarting the wait loop." >&2
  exec "$0" "$@"
fi

echo "[profile] GPU $GPU_INDEX is empty; launching all-model profiling."
echo "[profile] Config: $CONFIG_PATH"

profile_scope=(--enable-all)
for argument in "$@"; do
  if [[ "$argument" == "--models" || "$argument" == "--enable-all" ]]; then
    profile_scope=()
    break
  fi
done

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
DIFFUSION_HANDS_PYTHON="$PYTHON_BIN" \
  "$ROOT_DIR/scripts/profile_all_models.sh" \
    --config "$CONFIG_PATH" \
    "${profile_scope[@]}" \
    "$@"
