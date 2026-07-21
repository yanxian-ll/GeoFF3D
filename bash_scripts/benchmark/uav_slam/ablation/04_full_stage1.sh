#!/bin/bash
set -euo pipefail

# Training-stage ablation: keep the full method fixed and use the stage-1
# checkpoint. Compare this directly with 03_full, which uses stage-2 weights.
METHOD_NAME="04_full_stage1"
CHECKPOINT="${CHECKPOINT:-experiments/dom/uav_training/pi3x_zup_translation_yaw_aug_8v_6d_16ipg_2g/checkpoint-last.pth}"
SPATIAL_PARTITION="footprint_tree"
POST_CHUNK_ALIGN=1
POST_CHUNK_ALIGN_MODE="yaw_translation"
DEPTH_PRIOR="pred"
source "$(dirname "$0")/_base.sh"
