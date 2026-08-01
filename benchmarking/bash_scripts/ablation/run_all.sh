#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
RESULTS_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/experiments/benchmarking/ablation}"
for experiment in \
  00_temporal_base \
  01_base \
  02_yaw_before_propagation \
  02_propagation_before_yaw \
  03_full \
  04_full_stage1 \
  05_full_stronger_noise \
  06_full_translation_only \
  07_full_chunk20 \
  08_full_chunk40 \
  09_colmap_dense; do
  OUTPUT_ROOT="$RESULTS_ROOT" "$SCRIPT_DIR/${experiment}.sh" "$@"
done

python3 "$SCRIPT_DIR/summarize_seam_error.py" \
  --results-root "$RESULTS_ROOT"
