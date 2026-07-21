#!/bin/bash
set -euo pipefail

METHOD_NAME="mast3r_sfm"
OPTIM_RUNNER="mast3r_sfm"
OPTIM_METHOD="mast3r-sfm"
# METHOD_SCENE_LIST="${METHOD_SCENE_LIST:-$(dirname "$0")/../default_scenes.yaml}"
METHOD_SCENE_LIST="${METHOD_SCENE_LIST:-$(dirname "$0")/vggt_slam_scenes.yaml}"


MAX_SIDE="${MAX_SIDE:-512}"
PATCH_SIZE="${PATCH_SIZE:-16}"

MAST3R_PYTHON="${MAST3R_PYTHON:-python3}"
MAST3R_MODEL_PATH="${MAST3R_MODEL_PATH:-checkpoints/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
MAST3R_RETRIEVAL_MODEL="${MAST3R_RETRIEVAL_MODEL:-checkpoints/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth}"
MAST3R_SCENE_GRAPH="${MAST3R_SCENE_GRAPH:-retrieval-20-1}"

source "$(dirname "$0")/_run_optim_method.sh"
