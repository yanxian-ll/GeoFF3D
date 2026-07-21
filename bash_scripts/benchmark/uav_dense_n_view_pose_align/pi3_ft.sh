#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Benchmark Pi3 on UAV dense n-view benchmark with pose-based alignment.
#
# Usage:
#   bash bash_scripts/benchmark/uav_dense_n_view/pi3.sh [cuda_device]

set -euo pipefail

export HYDRA_FULL_ERROR=1
export NUMEXPR_MAX_THREADS=16

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CUDA_DEVICE="${1:-0}"

CHECKPOINT_PATH=${CHECKPOINT_PATH:-'${root_experiments_dir}/mapanything/uav_training/pi3x_finetuning_16v_6d_16ipg_2g_mvs/checkpoint-best.pth'}

MODEL_NAME="pi3"
OUTPUT_NAME="pi3_ft"
ALIGNMENT_MODE="pose"

# batch size, views, dataset, seed
batch_sizes_and_views=(
    # "4 8 benchmark_518_uavff3d_usegeo 8"
    "3 16 benchmark_518_uavff3d_usegeo 16"
    # "2 24 benchmark_518_uavff3d_usegeo 24"
    # "1 32 benchmark_518_uavff3d_usegeo 32"
)

cd "$ROOT_DIR"

echo "============================================"
echo "UAV Dense N-View Benchmark (Pose Align)"
echo "Model:          ${MODEL_NAME}"
echo "Alignment mode: ${ALIGNMENT_MODE}"
echo "CUDA device:    ${CUDA_DEVICE}"
echo "============================================"

for combo in "${batch_sizes_and_views[@]}"; do
    read -r batch_size num_views dataset seed <<< "$combo"

    echo ""
    echo ">>> Running ${MODEL_NAME}: dataset=${dataset}, num_views=${num_views}, batch_size=${batch_size}, seed=${seed}, alignment=${ALIGNMENT_MODE}"

    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python3 \
        benchmarking/dense_n_view/benchmark.py \
        machine=aws \
        seed="${seed}" \
        compute_abs_metrics=true \
        save_n_fused_ply=3 \
        +alignment_mode="${ALIGNMENT_MODE}" \
        dataset="${dataset}" \
        dataset.num_workers=12 \
        dataset.num_views="${num_views}" \
        batch_size="${batch_size}" \
        model="${MODEL_NAME}" \
        model.pretrained="${CHECKPOINT_PATH}" \
        hydra.run.dir='${root_experiments_dir}/pose_align/benchmarking/uav_dense_'"${num_views}"'_view/'"${OUTPUT_NAME}"

    echo "<<< Finished ${MODEL_NAME}: dataset=${dataset}, num_views=${num_views}, alignment=${ALIGNMENT_MODE}"
done

echo ""
echo "============================================"
echo "All UAV dense n-view pose-align runs completed: ${MODEL_NAME}"
echo "============================================"
