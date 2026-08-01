#!/bin/bash
set -euo pipefail

# Prior-input ablation: keep the full pipeline and mild pose noise, but provide
# only the translation prior to the model (no input rotation prior).
METHOD_NAME="06_full_translation_only"
SPATIAL_PARTITION="footprint_tree"
POST_CHUNK_ALIGN=true
POST_CHUNK_ALIGN_MODE="yaw_translation"
DEPTH_PRIOR="pred"
TRANSLATION_PRIOR="${TRANSLATION_PRIOR:-input}"
ROTATION_PRIOR="${ROTATION_PRIOR:-none}"

source "$(dirname "$0")/_base.sh"
