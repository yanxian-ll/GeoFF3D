#!/bin/bash
set -euo pipefail

METHOD_NAME="vggt_slam_sim3_ft"
OPTIM_RUNNER="vggt_slam"
OPTIM_METHOD="vggt-slam-sim3"
METHOD_SCENE_LIST="${METHOD_SCENE_LIST:-$(dirname "$0")/vggt_slam_scenes.yaml}"

VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-experiments/mapanything/uav_training/vggt_finetuning_16v_6d_16ipg_2g/checkpoint-best.pth}"
MAX_SIDE="${MAX_SIDE:-518}"
PATCH_SIZE="${PATCH_SIZE:-14}"
SUBMAP_SIZE="${SUBMAP_SIZE:-16}"
OVERLAPPING_WINDOW_SIZE="${OVERLAPPING_WINDOW_SIZE:-2}"

USE_SIM3="${USE_SIM3:-1}"
VGGT_SLAM_PYTHON="${VGGT_SLAM_PYTHON:-/opt/conda/envs/mapanything/bin/python}"

MAX_POINTS_PER_VIEW="${MAX_POINTS_PER_VIEW:-500000}"
VOXEL_DOWNSAMPLE="${VOXEL_DOWNSAMPLE:-0.01}"
POINT_DOWNSAMPLE="${POINT_DOWNSAMPLE:-1}"
GLOBAL_POINT_STRIDE="${GLOBAL_POINT_STRIDE:-1}"
HEADLESS="${HEADLESS:-1}"
KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-0}"

source "$(dirname "$0")/_run_optim_method.sh"
