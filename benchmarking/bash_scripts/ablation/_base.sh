#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

METHOD_NAME="${METHOD_NAME:?Set METHOD_NAME before sourcing _base.sh}"
SCENE_LIST="${SCENE_LIST:-$SCRIPT_DIR/ablation_scenes.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/experiments/benchmarking/ablation}"
CHECKPOINT="${CHECKPOINT:-}"
CHECKPOINT_PROFILE="${CHECKPOINT_PROFILE:-stage2}"

ALIGN="${ALIGN:-scale_yaw_translation}"
SPATIAL_PARTITION="${SPATIAL_PARTITION:-footprint_tree}"
FOOTPRINT_ESTIMATION="${FOOTPRINT_ESTIMATION:-sequential}"
CHUNK_ORDER="${CHUNK_ORDER:-spatial_center_bfs}"
TEMPORAL_OVERLAP_RATIO="${TEMPORAL_OVERLAP_RATIO:-0.25}"
MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-30}"
MIN_CHUNK_SIZE="${MIN_CHUNK_SIZE:-8}"

DEPTH_PRIOR="${DEPTH_PRIOR:-pred}"
TRANSLATION_PRIOR="${TRANSLATION_PRIOR:-input}"
ROTATION_PRIOR="${ROTATION_PRIOR:-input}"
RAY_PRIOR="${RAY_PRIOR:-pred}"

POST_CHUNK_ALIGN="${POST_CHUNK_ALIGN:-true}"
POST_CHUNK_ALIGN_MODE="${POST_CHUNK_ALIGN_MODE:-yaw_translation}"

POSE_PERTURB="${POSE_PERTURB:-true}"
POSE_PERTURB_XY_STD="${POSE_PERTURB_XY_STD:-0.5}"
POSE_PERTURB_Z_STD="${POSE_PERTURB_Z_STD:-0.8}"
POSE_PERTURB_YAW_STD_DEG="${POSE_PERTURB_YAW_STD_DEG:-1.0}"
POSE_PERTURB_XY_MAX="${POSE_PERTURB_XY_MAX:-2.0}"
POSE_PERTURB_Z_MAX="${POSE_PERTURB_Z_MAX:-2.0}"
POSE_PERTURB_YAW_MAX_DEG="${POSE_PERTURB_YAW_MAX_DEG:-3.0}"
POSE_PERTURB_SEED_OFFSET="${POSE_PERTURB_SEED_OFFSET:-930001}"

if [[ -n "$CHECKPOINT" ]]; then
  export CHECKPOINT
fi

python3 "$ROOT_DIR/benchmarking/bash_scripts/ours/run_all.py" --methods geoff3d \
  --checkpoint-profile "$CHECKPOINT_PROFILE" \
  --scene-list "$SCENE_LIST" \
  --output-root "$OUTPUT_ROOT" \
  --output-name "$METHOD_NAME" \
  "$@" \
  spatial_partition="$SPATIAL_PARTITION" \
  footprint_estimation="$FOOTPRINT_ESTIMATION" \
  chunk_order="$CHUNK_ORDER" \
  temporal_overlap_ratio="$TEMPORAL_OVERLAP_RATIO" \
  max_chunk_size="$MAX_CHUNK_SIZE" \
  min_chunk_size="$MIN_CHUNK_SIZE" \
  align="$ALIGN" \
  depth_prior="$DEPTH_PRIOR" \
  translation_prior="$TRANSLATION_PRIOR" \
  rotation_prior="$ROTATION_PRIOR" \
  ray_prior="$RAY_PRIOR" \
  post_chunk_align="$POST_CHUNK_ALIGN" \
  post_chunk_align_mode="$POST_CHUNK_ALIGN_MODE" \
  pose_perturb="$POSE_PERTURB" \
  pose_perturb_xy_std="$POSE_PERTURB_XY_STD" \
  pose_perturb_z_std="$POSE_PERTURB_Z_STD" \
  pose_perturb_yaw_std_deg="$POSE_PERTURB_YAW_STD_DEG" \
  pose_perturb_xy_max="$POSE_PERTURB_XY_MAX" \
  pose_perturb_z_max="$POSE_PERTURB_Z_MAX" \
  pose_perturb_yaw_max_deg="$POSE_PERTURB_YAW_MAX_DEG" \
  pose_perturb_seed_offset="$POSE_PERTURB_SEED_OFFSET"
