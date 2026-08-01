#!/bin/bash
set -euo pipefail

# Chunk-size ablation: full method with at most 40 views per chunk.
METHOD_NAME="08_full_chunk40"
SPATIAL_PARTITION="footprint_tree"
POST_CHUNK_ALIGN=true
POST_CHUNK_ALIGN_MODE="yaw_translation"
DEPTH_PRIOR="pred"
MAX_CHUNK_SIZE="${MAX_CHUNK_SIZE:-40}"

source "$(dirname "$0")/_base.sh"
