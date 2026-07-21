#!/bin/bash
set -euo pipefail

METHOD_NAME="on_the_fly_nvs"
OPTIM_RUNNER="gaussian"
OPTIM_METHOD="on-the-fly-nvs"
METHOD_SCENE_LIST="${METHOD_SCENE_LIST:-$(dirname "$0")/../default_scenes.yaml}"

PATCH_SIZE="${PATCH_SIZE:-14}"
MAX_SIDE="${MAX_SIDE:-518}"

source "$(dirname "$0")/_run_optim_method.sh"
