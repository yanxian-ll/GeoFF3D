#!/bin/bash
set -euo pipefail

METHOD_NAME="vggt_long"
OPTIM_RUNNER="vggt_long"
OPTIM_METHOD="vggt-long"
# METHOD_SCENE_LIST="${METHOD_SCENE_LIST:-$(dirname "$0")/../default_scenes.yaml}"
METHOD_SCENE_LIST="${METHOD_SCENE_LIST:-$(dirname "$0")/vggt_slam_scenes.yaml}"


PATCH_SIZE="${PATCH_SIZE:-14}"
MAX_SIDE="${MAX_SIDE:-518}"

VGGT_LONG_CONFIG="${VGGT_LONG_CONFIG:-third_party/vggt-long/configs/ours.yaml}"

source "$(dirname "$0")/_run_optim_method.sh"