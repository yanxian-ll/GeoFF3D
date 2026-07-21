#!/bin/bash
set -euo pipefail

METHOD_NAME="artdeco"
OPTIM_RUNNER="gaussian"
OPTIM_METHOD="artdeco"
METHOD_SCENE_LIST="${METHOD_SCENE_LIST:-$(dirname "$0")/../default_scenes.yaml}"

PATCH_SIZE="${PATCH_SIZE:-14}"
MAX_SIDE="${MAX_SIDE:-518}"

ARTDECO_CONFIG="${ARTDECO_CONFIG:-}"
ARTDECO_CHECKPOINT="${ARTDECO_CHECKPOINT:-}"

source "$(dirname "$0")/_run_optim_method.sh"
