#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

METHOD_NAME="vggt_slam2.0"
OPTIM_RUNNER="vggt_slam2.0"
OPTIM_METHOD="vggt-slam2.0"
METHOD_SCENE_LIST="${METHOD_SCENE_LIST:-$SCRIPT_DIR/test_scenes.yaml}"
PARAMS_LIST="${PARAMS_LIST:-$ROOT_DIR/bash_scripts/benchmark/uav_slam/optim/default_params.yaml:$SCRIPT_DIR/default_params.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-$ROOT_DIR/outputs/efficiency_scalability/${METHOD_NAME}}"

MAX_SIDE="${MAX_SIDE:-518}"
PATCH_SIZE="${PATCH_SIZE:-14}"
SUBMAP_SIZE="${SUBMAP_SIZE:-16}"
OVERLAPPING_WINDOW_SIZE="${OVERLAPPING_WINDOW_SIZE:-2}"
VGGT_SLAM2_PYTHON="${VGGT_SLAM2_PYTHON:-/opt/conda/envs/mapanything/bin/python}"
HEADLESS=1
DISABLE_KEYFRAME_SELECTION="${DISABLE_KEYFRAME_SELECTION:-0}"
RUN_METRICS=0

source "$ROOT_DIR/bash_scripts/benchmark/uav_slam/optim/_run_optim_method.sh"
