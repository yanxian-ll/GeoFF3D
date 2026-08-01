#!/bin/bash
set -euo pipefail

# Noise robustness ablation: keep the full method fixed and double every
# standard deviation and clipping threshold relative to the default mild noise.
METHOD_NAME="05_full_stronger_noise"
SPATIAL_PARTITION="footprint_tree"
POST_CHUNK_ALIGN=true
POST_CHUNK_ALIGN_MODE="yaw_translation"
DEPTH_PRIOR="pred"

POSE_PERTURB="${POSE_PERTURB:-true}"
POSE_PERTURB_XY_STD="${POSE_PERTURB_XY_STD:-1.0}"
POSE_PERTURB_Z_STD="${POSE_PERTURB_Z_STD:-1.6}"
POSE_PERTURB_YAW_STD_DEG="${POSE_PERTURB_YAW_STD_DEG:-2.0}"
POSE_PERTURB_XY_MAX="${POSE_PERTURB_XY_MAX:-4.0}"
POSE_PERTURB_Z_MAX="${POSE_PERTURB_Z_MAX:-4.0}"
POSE_PERTURB_YAW_MAX_DEG="${POSE_PERTURB_YAW_MAX_DEG:-6.0}"
POSE_PERTURB_SEED_OFFSET="${POSE_PERTURB_SEED_OFFSET:-930001}"

source "$(dirname "$0")/_base.sh"
