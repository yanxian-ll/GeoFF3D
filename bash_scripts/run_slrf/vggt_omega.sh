#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SCENE_DIR="${1:?Usage: $0 SCENE_DIR [OUTPUT_PATH] [Hydra overrides...]}"
SCENE_NAME="$(basename "${SCENE_DIR%/}")"
OUTPUT_PATH="${2:-$ROOT_DIR/experiments/uav_slrf/vggt_omega/$SCENE_NAME}"
shift $(( $# >= 2 ? 2 : 1 ))

exec python3 "$ROOT_DIR/scripts/run_slrf.py" \
  model=vggt_omega \
  scene_dir="$SCENE_DIR" \
  checkpoint="${CHECKPOINT:-$ROOT_DIR/experiments/uav_training/vggt_omega_8v_4d_8ipg_2g/checkpoint-best.pth}" \
  output_path="$OUTPUT_PATH" \
  device="cuda:${CUDA_DEVICE:-0}" \
  max_image_size="${MAX_IMAGE_SIZE:-512}" \
  footprint_estimation="${FOOTPRINT_ESTIMATION:-prior}" \
  align=sim3 \
  pose_prior=none \
  translation_prior=none \
  rotation_prior=none \
  ray_prior=none \
  depth_prior=none \
  max_chunk_size="${MAX_CHUNK_SIZE:-30}" \
  min_chunk_size="${MIN_CHUNK_SIZE:-8}" \
  post_chunk_align="${POST_CHUNK_ALIGN:-true}" \
  post_chunk_align_mode="${POST_CHUNK_ALIGN_MODE:-rigid}" \
  "$@"
