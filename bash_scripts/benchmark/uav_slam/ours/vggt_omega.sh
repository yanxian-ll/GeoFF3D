#!/bin/bash
set -euo pipefail

METHOD_NAME="vggt_omega"
MODEL="vggt_omega"
CHECKPOINT="${CHECKPOINT:-checkpoints/vggt-omega/vggt_omega_1b_512.pt}"

NORM_TYPE="${NORM_TYPE:-identity}"
PATCH_SIZE="${PATCH_SIZE:-16}"

MODEL_FAMILY="no_prior"

ALIGN="pose_sim3"
RECENTER="none"

POSE_PRIOR="none"
TRANSLATION_PRIOR="none"
ROTATION_PRIOR="none"
RAY_PRIOR="none"
DEPTH_PRIOR="none"

MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-50}"
MIN_CHUNK_SIZE="${MIN_CHUNK_SIZE:-8}"

POST_CHUNK_ALIGN="${POST_CHUNK_ALIGN:-1}"
POST_CHUNK_ALIGN_MODE="${POST_CHUNK_ALIGN_MODE:-rigid}"

source "$(dirname "$0")/_run_spatial_method.sh"
