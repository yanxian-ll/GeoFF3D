#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
RESULTS_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/ablation}"
for experiment in \
  00_temporal_base \
  01_base \
  02_yaw_before_propagation \
  02_propagation_before_yaw \
  03_full \
  04_full_stage1; do
  OUTPUT_ROOT="$RESULTS_ROOT" "$SCRIPT_DIR/${experiment}.sh" "$@"
done

python3 "$SCRIPT_DIR/summarize_seam_error.py" \
  --results-root "$RESULTS_ROOT"
