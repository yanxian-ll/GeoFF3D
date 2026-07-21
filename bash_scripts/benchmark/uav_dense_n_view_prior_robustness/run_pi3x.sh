#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DEVICE="${1:-0}"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_ROOT="$(realpath -m "${2:-${OUTPUT_ROOT:-$ROOT_DIR/outputs/prior_robustness}}")"
SEEDS_STRING="${SEEDS:-16}"

run_values() {
  local condition="$1"
  shift
  for value in "$@"; do
    for seed in $SEEDS_STRING; do
      CONDITION="$condition" VALUE="$value" SEED="$seed" \
        CUDA_DEVICE="$CUDA_DEVICE" OUTPUT_ROOT="$OUTPUT_ROOT" \
        "$SCRIPT_DIR/run_pi3x_condition.sh"
    done
  done
}

run_values horizontal 0 0.5 1 2 5
run_values vertical 0 0.2 0.5 1 2
# Pi3X cannot use a single relative pose prior, so retained=1 is not run.
run_values missing 16 8 4 3 2
