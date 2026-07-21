#!/bin/bash
set -euo pipefail

METHOD_NAME="pi3x_ft"
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

source "$(dirname "$0")/_run_spatial_method.sh"
