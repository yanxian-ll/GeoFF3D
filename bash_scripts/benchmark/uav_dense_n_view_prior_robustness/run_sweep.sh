#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CUDA_DEVICE="${1:-0}"
OUTPUT_ROOT="$(realpath -m "${2:-${OUTPUT_ROOT:-$ROOT_DIR/outputs/prior_robustness_3seed}}")"
SEEDS_STRING="${SEEDS:-16 17 18}"
export OUTPUT_ROOT

echo "[INFO] Prior-robustness output root: $OUTPUT_ROOT"

run_values() {
  local condition="$1"
  shift
  for value in "$@"; do
    for seed in $SEEDS_STRING; do
      for alignment in prior_yaw prior_pose; do
        CONDITION="$condition" VALUE="$value" SEED="$seed" \
          ALIGNMENT_MODE="$alignment" CUDA_DEVICE="$CUDA_DEVICE" \
          OUTPUT_ROOT="$OUTPUT_ROOT" \
          "$SCRIPT_DIR/run_condition.sh"
      done
    done
  done
}

run_values horizontal 0 0.5 1 2 5
run_values vertical 0 0.2 0.5 1 2
run_values missing 16 8 4 3 2

SEEDS="$SEEDS_STRING" OUTPUT_ROOT="$OUTPUT_ROOT" \
  "$SCRIPT_DIR/run_pi3x.sh" "$CUDA_DEVICE" "$OUTPUT_ROOT"

python3 "$SCRIPT_DIR/plot_prior_robustness.py" \
  --results-root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/prior_robustness_abc" \
  --font-scale "${PLOT_FONT_SCALE:-1.0}" \
  --padding-ratio "${PLOT_PADDING_RATIO:-0.02}" \
  $( [[ "${PLOT_SHOW_LEGEND:-1}" == "0" ]] && echo --no-show-legend || echo --show-legend )
