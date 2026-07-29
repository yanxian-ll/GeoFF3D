#!/bin/bash
set -euo pipefail

METHOD_NAME="geoff3d"
MODEL="geoff3d"
# CHECKPOINT="${CHECKPOINT:-experiments/uav_training/geoff3d_stage1_geoff3d_12v_4d_12ipg_2g/checkpoint-last.pth}"
CHECKPOINT="${CHECKPOINT:-experiments/uav_training/geoff3d_8v_4d_8ipg_2g/checkpoint-best.pth}"



NORM_TYPE="${NORM_TYPE:-identity}"
PATCH_SIZE="${PATCH_SIZE:-14}"

MODEL_FAMILY="ours"

# 还是使用额外对齐，保证鲁棒性
ALIGN="${ALIGN:-pose_scale_yaw_translation}"
RECENTER="${RECENTER:-auto}"

TRANSLATION_PRIOR="${TRANSLATION_PRIOR:-input}"
ROTATION_PRIOR="${ROTATION_PRIOR:-input}"

# 有真实内参/深度时 auto=input；没有时 auto=pred
RAY_PRIOR="${RAY_PRIOR:-pred}"
DEPTH_PRIOR="${DEPTH_PRIOR:-pred}"

POSE_PRIOR="none"

MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-30}"
MIN_CHUNK_SIZE="${MIN_CHUNK_SIZE:-8}"

# Optional hierarchical post-alignment across chunks.
# Keep off by default; enable when GPS/translation priors are noisy.
POST_CHUNK_ALIGN="${POST_CHUNK_ALIGN:-1}"
POST_CHUNK_ALIGN_MODE="${POST_CHUNK_ALIGN_MODE:-yaw_translation}"

source "$(dirname "$0")/_run_spatial_method.sh"
