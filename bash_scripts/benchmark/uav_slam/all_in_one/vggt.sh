#!/bin/bash
set -euo pipefail

METHOD_NAME="vggt"
ALL_IN_ONE_RUNNER="model"
ALL_IN_ONE_METHOD="vggt"
# METHOD_SCENE_LIST="${METHOD_SCENE_LIST:-$(dirname "$0")/../default_scenes.yaml}"
METHOD_SCENE_LIST="${METHOD_SCENE_LIST:-$(dirname "$0")/vggt_pi3_scenes.yaml}"

MAX_SIDE="${MAX_SIDE:-518}"
PATCH_SIZE="${PATCH_SIZE:-14}"
NORM_TYPE="${NORM_TYPE:-identity}"
CHECKPOINT="${CHECKPOINT:-checkpoints/vggt}"
CONF_QUANTILE="${CONF_QUANTILE:-0.0}"

source "$(dirname "$0")/_run_all_in_one_method.sh"
