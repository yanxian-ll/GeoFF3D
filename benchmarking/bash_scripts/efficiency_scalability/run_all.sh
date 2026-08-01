#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
CUDA_DEVICE="${1:-0}"

"$SCRIPT_DIR/pi3x_world_translation.sh" "$CUDA_DEVICE"
"$SCRIPT_DIR/vggt_slam2.0.sh" "$CUDA_DEVICE"
"$SCRIPT_DIR/lingbot-map.sh" "$CUDA_DEVICE"

python3 "$SCRIPT_DIR/plot_scalability.py" \
  --results-root "$ROOT_DIR/experiments/benchmarking/efficiency_scalability" \
  --output "$ROOT_DIR/experiments/benchmarking/efficiency_scalability/efficiency_scalability" \
  --font-scale "${PLOT_FONT_SCALE:-1.2}" \
  --padding-ratio "${PLOT_PADDING_RATIO:-0.02}" \
  $( [[ "${PLOT_SHOW_LEGEND:-1}" == "0" ]] && echo --no-show-legend || echo --show-legend )
