#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

METHOD_NAME="${METHOD_NAME:?Set METHOD_NAME before sourcing _base.sh}"
MODEL="${MODEL:-geoff3d}"
CHECKPOINT="${CHECKPOINT:-experiments/dom/uav_training/pi3x_zup_translation_yaw_aug_8v_6d_16ipg_2g_stage2/checkpoint-best.pth}"
SCENE_LIST="${SCENE_LIST:-$SCRIPT_DIR/ablation_scenes.yaml}"
PARAMS_LIST="${PARAMS_LIST:-$ROOT_DIR/bash_scripts/benchmark/uav_slam/ours/default_params.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/ablation}"
OUTPUT_BASE="${OUTPUT_BASE:-$OUTPUT_ROOT/${METHOD_NAME}}"

MODEL_FAMILY="${MODEL_FAMILY:-ours}"
NORM_TYPE="${NORM_TYPE:-identity}"
PATCH_SIZE="${PATCH_SIZE:-14}"
# Per-chunk prediction-to-prior alignment.  Keep this fixed across the
# ablation; POST_CHUNK_ALIGN_MODE is the hierarchical residual alignment
# factor studied below.
ALIGN="${ALIGN:-pose_scale_yaw_translation}"
RECENTER="${RECENTER:-auto}"
POSE_PRIOR="${POSE_PRIOR:-none}"
TRANSLATION_PRIOR="${TRANSLATION_PRIOR:-input}"
ROTATION_PRIOR="${ROTATION_PRIOR:-input}"
RAY_PRIOR="${RAY_PRIOR:-pred}"
DEPTH_PRIOR="${DEPTH_PRIOR:-pred}"

MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-50}"
MIN_CHUNK_SIZE="${MIN_CHUNK_SIZE:-8}"
CHUNK_ORDER="${CHUNK_ORDER:-spatial_center_bfs}"
SPATIAL_PARTITION="${SPATIAL_PARTITION:-footprint_tree}"
POSE_GRID_SIZE="${POSE_GRID_SIZE:-0}"
POSE_GRID_NEIGHBOR_RADIUS="${POSE_GRID_NEIGHBOR_RADIUS:-1}"

POST_CHUNK_ALIGN="${POST_CHUNK_ALIGN:-1}"
POST_CHUNK_ALIGN_MODE="${POST_CHUNK_ALIGN_MODE:-yaw_translation}"
RUN_METRICS="${RUN_METRICS:-1}"
COMPUTE_SEAM_ERROR="${COMPUTE_SEAM_ERROR:-1}"
SEAM_ERROR_MAX_POINTS_PER_EDGE="${SEAM_ERROR_MAX_POINTS_PER_EDGE:-20000}"
LOG_CHUNKS="${LOG_CHUNKS:-1}"
KEEP_CHUNK_CACHE="${KEEP_CHUNK_CACHE:-1}"

source "$ROOT_DIR/bash_scripts/benchmark/uav_slam/ours/_run_spatial_method.sh"
