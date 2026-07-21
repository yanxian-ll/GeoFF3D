#!/bin/bash
set -euo pipefail

METHOD_NAME="streamvggt"
OPTIM_METHOD="streamvggt"

METHOD_SCENE_LIST="${METHOD_SCENE_LIST:-$(dirname "$0")/stream_scenes.yaml}"

# Use a separate interpreter/env for StreamVGGT to avoid mutating mapanything.
STREAM_PYTHON="${STREAM_PYTHON:-${STREAMVGGT_PYTHON:-/opt/conda/envs/streamvggt/bin/python}}"
STREAM_MODEL_PATH="${STREAM_MODEL_PATH:-${STREAMVGGT_MODEL_PATH:-}}"

source "$(dirname "$0")/_run_stream_method.sh"
