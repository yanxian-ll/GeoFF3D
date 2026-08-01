#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SCENE_DIR="${1:?Usage: $0 SCENE_DIR [OUTPUT_PATH] [Hydra overrides...]}"
SCENE_NAME="$(basename "${SCENE_DIR%/}")"
OUTPUT_PATH="${2:-$ROOT_DIR/experiments/uav_slrf/geoff3d/$SCENE_NAME}"
shift $(( $# >= 2 ? 2 : 1 ))

exec python3 "$ROOT_DIR/scripts/run_slrf.py" \
  model=geoff3d \
  scene_dir="$SCENE_DIR" \
  checkpoint="${CHECKPOINT:-$ROOT_DIR/checkpoints/geoff3d/checkpoint-best.pth}" \
  output_path="$OUTPUT_PATH" \
  device="cuda:${CUDA_DEVICE:-0}" \
  footprint_estimation="${FOOTPRINT_ESTIMATION:-sequential}" \
  align="${ALIGN:-scale_yaw_translation}" \
  translation_prior=input \
  rotation_prior="${ROTATION_PRIOR:-input}" \
  ray_prior="${RAY_PRIOR:-pred}" \
  depth_prior="${DEPTH_PRIOR:-pred}" \
  max_chunk_size="${MAX_CHUNK_SIZE:-30}" \
  min_chunk_size="${MIN_CHUNK_SIZE:-8}" \
  post_chunk_align="${POST_CHUNK_ALIGN:-true}" \
  post_chunk_align_mode="${POST_CHUNK_ALIGN_MODE:-yaw_translation}" \
  "$@"
