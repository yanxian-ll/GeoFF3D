#!/bin/bash

# Pose-aligned dense n-view evaluation for the Stage-2 model trained without
# GravityAlignmentLoss.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_PATH=${CHECKPOINT_PATH:-'${root_experiments_dir}/dom/uav_training/pi3x_zup_translation_yaw_aug_8v_6d_16ipg_2g_wogravity_stage2/checkpoint-best.pth'}
export CHECKPOINT_PATH
export RUN_TAG="${RUN_TAG:-geoff3d_p_wogravity}"
export ALIGNMENT_MODE="${ALIGNMENT_MODE:-pose_yaw}"
export BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-benchmarking/dense_n_view/benchmark_absolute_world.py}"

exec bash "$SCRIPT_DIR/pi3x_transup_p.sh" "$@"
