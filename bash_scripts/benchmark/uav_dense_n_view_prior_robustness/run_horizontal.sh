#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DEVICE="${1:-0}"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_ROOT="$(realpath -m "${2:-${OUTPUT_ROOT:-$ROOT_DIR/outputs/prior_robustness}}")"
for value in 0 0.5 1 2 5; do
  for seed in ${SEEDS:-16 17 18}; do
    for alignment in prior_yaw prior_pose; do
      CONDITION=horizontal VALUE="$value" SEED="$seed" ALIGNMENT_MODE="$alignment" \
        CUDA_DEVICE="$CUDA_DEVICE" OUTPUT_ROOT="$OUTPUT_ROOT" \
        "$SCRIPT_DIR/run_condition.sh"
    done
  done
done
