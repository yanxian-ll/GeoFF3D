#!/bin/bash
set -euo pipefail

METHOD_NAME="pi3x_ft_gnss_perturb"
MODEL="pi3x"
CHECKPOINT="${CHECKPOINT:-experiments/mapanything/uav_training/pi3x_finetuning_16v_6d_16ipg_2g_mvs/checkpoint-best.pth}"

NORM_TYPE="${NORM_TYPE:-identity}"
PATCH_SIZE="${PATCH_SIZE:-14}"

MODEL_FAMILY="input_prior"

ALIGN="${ALIGN:-pose_sim3}"
RECENTER="none"

POSE_PRIOR="${POSE_PRIOR:-input}"
RAY_PRIOR="${RAY_PRIOR:-pred}"
DEPTH_PRIOR="${DEPTH_PRIOR:-pred}"

TRANSLATION_PRIOR="none"
ROTATION_PRIOR="none"

MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-50}"
MIN_CHUNK_SIZE="${MIN_CHUNK_SIZE:-8}"

POST_CHUNK_ALIGN="${POST_CHUNK_ALIGN:-1}"
POST_CHUNK_ALIGN_MODE="${POST_CHUNK_ALIGN_MODE:-rigid}"

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
