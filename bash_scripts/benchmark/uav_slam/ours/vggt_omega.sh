#!/bin/bash
set -euo pipefail

METHOD_NAME="vggt_omega_ft"
MODEL="vggt_omega"
CHECKPOINT="${CHECKPOINT:-experiments/uav_training/vggt_omega_8v_4d_8ipg_2g/checkpoint-best.pth}"

# The fine-tuned checkpoint is complete, so do not load the official base
# VGGT-Omega weights before applying it.
LOAD_PRETRAINED_WEIGHTS="${LOAD_PRETRAINED_WEIGHTS:-0}"

NORM_TYPE="${NORM_TYPE:-identity}"
PATCH_SIZE="${PATCH_SIZE:-16}"
MAX_SIDE="${MAX_SIDE:-512}"

MODEL_FAMILY="no_prior"

ALIGN="pose_sim3"
RECENTER="none"

POSE_PRIOR="none"
TRANSLATION_PRIOR="none"
ROTATION_PRIOR="none"
RAY_PRIOR="none"
DEPTH_PRIOR="none"

MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-30}"
MIN_CHUNK_SIZE="${MIN_CHUNK_SIZE:-8}"

POST_CHUNK_ALIGN="${POST_CHUNK_ALIGN:-1}"
POST_CHUNK_ALIGN_MODE="${POST_CHUNK_ALIGN_MODE:-rigid}"

source "$(dirname "$0")/_run_spatial_method.sh"
