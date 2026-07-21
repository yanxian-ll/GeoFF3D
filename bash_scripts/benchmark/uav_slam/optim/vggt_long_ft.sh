#!/bin/bash
set -euo pipefail

METHOD_NAME="vggt_long_ft"
OPTIM_RUNNER="vggt_long"
OPTIM_METHOD="vggt-long"
METHOD_SCENE_LIST="${METHOD_SCENE_LIST:-$(dirname "$0")/vggt_slam_scenes.yaml}"

VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-experiments/mapanything/uav_training/vggt_finetuning_16v_6d_16ipg_2g/checkpoint-best.pth}"
PATCH_SIZE="${PATCH_SIZE:-14}"
MAX_SIDE="${MAX_SIDE:-518}"
VGGT_LONG_CONFIG="${VGGT_LONG_CONFIG:-third_party/vggt-long/configs/ours.yaml}"

source "$(dirname "$0")/_run_optim_method.sh"
