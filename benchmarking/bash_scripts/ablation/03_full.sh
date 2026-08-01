#!/bin/bash
set -euo pipefail

# Both branches meet here: tree + post yaw alignment + prior propagation.
METHOD_NAME="03_full"
SPATIAL_PARTITION="footprint_tree"
POST_CHUNK_ALIGN=true
POST_CHUNK_ALIGN_MODE="yaw_translation"
DEPTH_PRIOR="pred"
source "$(dirname "$0")/_base.sh"
