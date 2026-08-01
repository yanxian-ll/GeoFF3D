#!/bin/bash
set -euo pipefail

# Training-stage ablation: keep the full method fixed and use the stage-1
# checkpoint. Compare this directly with 03_full, which uses stage-2 weights.
METHOD_NAME="04_full_stage1"
CHECKPOINT_PROFILE="stage1"
SPATIAL_PARTITION="footprint_tree"
POST_CHUNK_ALIGN=true
POST_CHUNK_ALIGN_MODE="yaw_translation"
DEPTH_PRIOR="pred"
source "$(dirname "$0")/_base.sh"
