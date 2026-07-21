#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

METHOD_NAME="lingbot-map"
OPTIM_RUNNER="streaming"
OPTIM_METHOD="lingbot-map"
METHOD_SCENE_LIST="${METHOD_SCENE_LIST:-$SCRIPT_DIR/test_scenes.yaml}"
PARAMS_LIST="${PARAMS_LIST:-$ROOT_DIR/bash_scripts/benchmark/uav_slam/stream/default_params.yaml:$SCRIPT_DIR/default_params.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-$ROOT_DIR/outputs/efficiency_scalability/${METHOD_NAME}}"

LINGBOT_STREAM_MODE="${LINGBOT_STREAM_MODE:-streaming}"
LINGBOT_KEYFRAME_INTERVAL="${LINGBOT_KEYFRAME_INTERVAL:-auto}"
USE_SDPA="${USE_SDPA:-1}"
RUN_METRICS=0

source "$ROOT_DIR/bash_scripts/benchmark/uav_slam/optim/_run_optim_method.sh"
