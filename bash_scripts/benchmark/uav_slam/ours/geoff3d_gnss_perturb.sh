#!/bin/bash
set -euo pipefail

METHOD_NAME="geoff3d_gnss_perturb"
MODEL="geoff3d"
CHECKPOINT="${CHECKPOINT:-${STAGE2_CHECKPOINT:-experiments/uav_training/geoff3d_8v_4d_8ipg_2g/checkpoint-best.pth}}"

# A complete GeoFF3D checkpoint already contains the model weights. Avoid
# loading the 5+ GB base Pi3X weights before applying the checkpoint.
LOAD_PRETRAINED_WEIGHTS="${LOAD_PRETRAINED_WEIGHTS:-0}"

NORM_TYPE="${NORM_TYPE:-identity}"
PATCH_SIZE="${PATCH_SIZE:-14}"

MODEL_FAMILY="ours"

ALIGN="${ALIGN:-pose_scale_yaw_translation}"
RECENTER="${RECENTER:-auto}"

TRANSLATION_PRIOR="${TRANSLATION_PRIOR:-input}"
ROTATION_PRIOR="${ROTATION_PRIOR:-input}"

RAY_PRIOR="${RAY_PRIOR:-pred}"
DEPTH_PRIOR="${DEPTH_PRIOR:-pred}"

POSE_PRIOR="none"

MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-30}"
MIN_CHUNK_SIZE="${MIN_CHUNK_SIZE:-8}"

POST_CHUNK_ALIGN="${POST_CHUNK_ALIGN:-1}"
POST_CHUNK_ALIGN_MODE="${POST_CHUNK_ALIGN_MODE:-yaw_translation}"

# Approximate single-frequency UAV GNSS prior noise. Defaults are intentionally
# mild enough to remain plausible for consumer/prosumer drone metadata.
POSE_PERTURB="${POSE_PERTURB:-1}"
POSE_PERTURB_XY_STD="${POSE_PERTURB_XY_STD:-0.5}"
POSE_PERTURB_Z_STD="${POSE_PERTURB_Z_STD:-0.8}"
POSE_PERTURB_YAW_STD_DEG="${POSE_PERTURB_YAW_STD_DEG:-1.0}"
POSE_PERTURB_XY_MAX="${POSE_PERTURB_XY_MAX:-2.0}"
POSE_PERTURB_Z_MAX="${POSE_PERTURB_Z_MAX:-2.0}"
POSE_PERTURB_YAW_MAX_DEG="${POSE_PERTURB_YAW_MAX_DEG:-3.0}"
POSE_PERTURB_SEED_OFFSET="${POSE_PERTURB_SEED_OFFSET:-930001}"

source "$(dirname "$0")/_run_spatial_method.sh"
