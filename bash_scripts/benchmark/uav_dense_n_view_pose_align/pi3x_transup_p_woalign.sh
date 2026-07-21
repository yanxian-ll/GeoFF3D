#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Benchmark GeoFF3D on UAV dense n-view benchmark
# using the absolute-world evaluator.
#
# This script does NOT convert GT/Pred to ref0.
# Default alignment is none.
#
# Usage:
#   bash bash_scripts/benchmark/uav_dense_n_view_pose_align/pi3x_transup_cp_woalign.sh [cuda_device]
#
# Override alignment mode:
#   ALIGNMENT_MODE=none   bash bash_scripts/benchmark/uav_dense_n_view_pose_align/pi3x_transup_cp_woalign.sh 0
#   ALIGNMENT_MODE=points bash bash_scripts/benchmark/uav_dense_n_view_pose_align/pi3x_transup_cp_woalign.sh 0
#   ALIGNMENT_MODE=pose   bash bash_scripts/benchmark/uav_dense_n_view_pose_align/pi3x_transup_cp_woalign.sh 0

set -euo pipefail

export HYDRA_FULL_ERROR=1
export NUMEXPR_MAX_THREADS=16

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CUDA_DEVICE="${1:-0}"

MODEL_NAME="geoff3d"
ALIGNMENT_MODE="${ALIGNMENT_MODE:-none}"

# GeoFF3D calibration+pose benchmark:
#   image + input calibration/ray prior + input camera pose translation/rotation
#   no depth prior

CHECKPOINT_PATH=${CHECKPOINT_PATH:-'${root_experiments_dir}/dom/uav_training/pi3x_zup_translation_yaw_aug_8v_6d_16ipg_2g_stage2/checkpoint-best.pth'}

RUN_TAG=${RUN_TAG:-geoff3d_p_woalign}

OUTPUT_NAME="${RUN_TAG}"

# batch size, views, dataset, seed
batch_sizes_and_views=(
    # "4 8 benchmark_518_uavff3d_usegeo 8"
    "3 16 benchmark_518_uavff3d_usegeo 16"
    # "2 24 benchmark_518_uavff3d_usegeo 24"
    # "1 32 benchmark_518_uavff3d_usegeo 32"
)

cd "$ROOT_DIR"

echo "============================================"
echo "UAV Dense N-View Benchmark (Absolute World)"
echo "Model:       ${MODEL_NAME}"
echo "Alignment mode: ${ALIGNMENT_MODE}"
echo "CUDA device: ${CUDA_DEVICE}"
echo "============================================"

for combo in "${batch_sizes_and_views[@]}"; do
    read -r batch_size num_views dataset seed <<< "$combo"

    echo ""
    echo ">>> Running ${MODEL_NAME}: dataset=${dataset}, num_views=${num_views}, batch_size=${batch_size}, seed=${seed}"

    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python3 \
        benchmarking/dense_n_view/benchmark_absolute_world.py \
        machine=aws \
        seed="${seed}" \
        compute_abs_metrics=true \
        save_n_fused_ply=3 \
        +alignment_mode="${ALIGNMENT_MODE}" \
        +filter_abs_failures=true \
        +max_abs_pose_ate=200.0 \
        +max_abs_pointmap_rmse=200.0 \
        +max_abs_chamfer=200.0 \
        dataset="${dataset}" \
        dataset.principal_point_centered=true \
        dataset.num_workers=12 \
        dataset.num_views="${num_views}" \
        batch_size="${batch_size}" \
        model="${MODEL_NAME}" \
        model/task=world_translation_prior \
        model.task.overall_prob=1.0 \
        model.task.ray_dirs_prob=0.0 \
        model.task.depth_prob=0.0 \
        model.task.sparse_depth_prob=0.0 \
        model.task.world_translation_prob=1.0 \
        model.task.world_rotation_prob=1.0 \
        model.model_config.use_world_translation_prior=true \
        model.model_config.use_world_rotation_prior=true \
        model.model_config.force_rotation_prior_for_degenerate_translation=false \
        model.model_config.translation_normalization=mean \
        model.model_config.de_normalize_outputs=true \
        model.pretrained="${CHECKPOINT_PATH}" \
        hydra.run.dir='${root_experiments_dir}/pose_align/benchmarking/uav_dense_'"${num_views}"'_view/'"${OUTPUT_NAME}"

    echo "<<< Finished ${MODEL_NAME}: dataset=${dataset}, num_views=${num_views}"
done

echo ""
echo "============================================"
echo "All UAV dense n-view absolute-world runs completed: ${MODEL_NAME}"
echo "============================================"
