#!/bin/bash
set -euo pipefail

METHOD_NAME="ttt3r"
OPTIM_METHOD="ttt3r"

METHOD_SCENE_LIST="${METHOD_SCENE_LIST:-$(dirname "$0")/stream_scenes.yaml}"

# Use a separate interpreter/env for TTT3R to avoid mutating mapanything.
STREAM_PYTHON="${STREAM_PYTHON:-${TTT3R_PYTHON:-/opt/conda/envs/ttt3r/bin/python}}"
STREAM_MODEL_PATH="${STREAM_MODEL_PATH:-${TTT3R_MODEL_PATH:-checkpoints/ttt3r/cut3r_512_dpt_4_64.pth}}"

source "$(dirname "$0")/_run_stream_method.sh"
