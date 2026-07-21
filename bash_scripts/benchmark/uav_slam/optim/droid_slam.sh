#!/bin/bash
set -euo pipefail

METHOD_NAME="droid_slam"
OPTIM_RUNNER="droid_slam"
OPTIM_METHOD="droid-slam"

# Reuse the same scene list as other optimization baselines unless overridden.
METHOD_SCENE_LIST="${METHOD_SCENE_LIST:-$(dirname "$0")/vggt_slam_scenes.yaml}"

# DROID-SLAM internally resizes again to roughly 384x512, but prepared images
# keep intrinsics consistent and bound IO size before entering the third-party
# code.
MAX_SIDE="${MAX_SIDE:-518}"
SIZE_MULTIPLE="${SIZE_MULTIPLE:-8}"

DROID_PYTHON="${DROID_PYTHON:-python3}"
DROID_ROOT="${DROID_ROOT:-third_party/DROID-SLAM}"
DROID_WEIGHTS="${DROID_WEIGHTS:-checkpoints/droid/droid.pth}"
DROID_BUFFER="${DROID_BUFFER:-1024}"
DROID_UPSAMPLE="${DROID_UPSAMPLE:-1}"
DROID_ASYNCHRONOUS="${DROID_ASYNCHRONOUS:-0}"

MAX_PRED_POINTS="${MAX_PRED_POINTS:-800000}"
MAX_GT_POINTS="${MAX_GT_POINTS:-800000}"

source "$(dirname "$0")/_run_optim_method.sh"
