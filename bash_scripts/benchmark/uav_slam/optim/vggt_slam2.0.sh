#!/bin/bash
set -euo pipefail

METHOD_NAME="vggt_slam2.0"
OPTIM_RUNNER="vggt_slam2.0"
OPTIM_METHOD="vggt-slam2.0"

# Reuse vggt_slam scene list.
METHOD_SCENE_LIST="${METHOD_SCENE_LIST:-$(dirname "$0")/vggt_slam_scenes.yaml}"

MAX_SIDE="${MAX_SIDE:-518}"
PATCH_SIZE="${PATCH_SIZE:-14}"
SUBMAP_SIZE="${SUBMAP_SIZE:-16}"
OVERLAPPING_WINDOW_SIZE="${OVERLAPPING_WINDOW_SIZE:-2}"
VGGT_SLAM2_PYTHON="${VGGT_SLAM2_PYTHON:-/opt/conda/envs/mapanything/bin/python}"

# Match ours/geoff3d output policy.
MAX_POINTS_PER_VIEW="${MAX_POINTS_PER_VIEW:-500000}"
MAX_GT_POINTS="${MAX_GT_POINTS:-800000}"
VOXEL_DOWNSAMPLE="${VOXEL_DOWNSAMPLE:-0.01}"
POINT_DOWNSAMPLE="${POINT_DOWNSAMPLE:-1}"
GLOBAL_POINT_STRIDE="${GLOBAL_POINT_STRIDE:-1}"

# VGGT-SLAM 2.0 benchmark should not try to open a viewer.
HEADLESS="${HEADLESS:-1}"

# Keep VGGT-SLAM 2.0 default keyframe selection unless explicitly disabled.
DISABLE_KEYFRAME_SELECTION="${DISABLE_KEYFRAME_SELECTION:-0}"

KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-0}"

source "$(dirname "$0")/_run_optim_method.sh"
