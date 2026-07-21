#!/bin/bash
set -euo pipefail

METHOD_NAME="lingbot-map"
OPTIM_METHOD="lingbot-map"

# METHOD_SCENE_LIST="${METHOD_SCENE_LIST:-$(dirname "$0")/../default_scenes.yaml}"
METHOD_SCENE_LIST="${METHOD_SCENE_LIST:-$(dirname "$0")/stream_scenes.yaml}"

# Required unless the wrapper is allowed to run with random/unloaded weights.
# LINGBOT_MODEL_PATH=/path/to/lingbot-map.pt
LINGBOT_STREAM_MODE="${LINGBOT_STREAM_MODE:-streaming}"
LINGBOT_KEYFRAME_INTERVAL="${LINGBOT_KEYFRAME_INTERVAL:-auto}"
USE_SDPA="${USE_SDPA:-1}"

source "$(dirname "$0")/_run_stream_method.sh"
