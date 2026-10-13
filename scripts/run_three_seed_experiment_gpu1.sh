#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# configs/experiment.yaml fixes physical GPU 1 and seeds 0, 1, and 2.
exec "$ROOT_DIR/scripts/run_all_models.sh" "$ROOT_DIR/configs/experiment.yaml"
