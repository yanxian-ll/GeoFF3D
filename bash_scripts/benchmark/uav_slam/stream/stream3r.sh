#!/bin/bash
set -euo pipefail

METHOD_NAME="stream3r"
OPTIM_METHOD="stream3r"

METHOD_SCENE_LIST="${METHOD_SCENE_LIST:-$(dirname "$0")/stream_scenes.yaml}"

# Use a separate interpreter/env for STream3R to avoid mutating mapanything.
STREAM_PYTHON="${STREAM_PYTHON:-${STREAM3R_PYTHON:-/opt/conda/envs/stream3r/bin/python}}"
STREAM_MODEL_PATH="${STREAM_MODEL_PATH:-${STREAM3R_MODEL_PATH:-yslan/STream3R}}"
STREAM_MODE="${STREAM_MODE:-${STREAM3R_MODE:-causal}}"

source "$(dirname "$0")/_run_stream_method.sh"
