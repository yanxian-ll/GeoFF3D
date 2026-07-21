#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Benchmark MapAnything on UAV dense n-view benchmark (pose align).
#
# Usage:
#   bash bash_scripts/benchmark/uav_dense_n_view/mapa_cp.sh [cuda_device]
#
# Override alignment mode:
#   ALIGNMENT_MODE=points bash bash_scripts/benchmark/uav_dense_n_view/mapa_cp.sh 0
#   ALIGNMENT_MODE=none   bash bash_scripts/benchmark/uav_dense_n_view/mapa_cp.sh 0

set -euo pipefail

export HYDRA_FULL_ERROR=1
export NUMEXPR_MAX_THREADS=16

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CUDA_DEVICE="${1:-0}"

MODEL_NAME="mapanything"
OUTPUT_NAME="mapa"
ALIGNMENT_MODE="${ALIGNMENT_MODE:-pose}"

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
echo "Model:       ${MODEL_NAME}"
echo "Alignment mode: ${ALIGNMENT_MODE}"
echo "CUDA device: ${CUDA_DEVICE}"
echo "============================================"

for combo in "${batch_sizes_and_views[@]}"; do
    read -r batch_size num_views dataset seed <<< "$combo"

    echo ""
    echo ">>> Running ${MODEL_NAME}: dataset=${dataset}, num_views=${num_views}, batch_size=${batch_size}, seed=${seed}"

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
        model/task=images_only \
        model.encoder.uses_torch_hub=false \
        model.pretrained='checkpoints/map-anything/map-anything.pth' \
        hydra.run.dir='${root_experiments_dir}/pose_align/benchmarking/uav_dense_'"${num_views}"'_view/'"${OUTPUT_NAME}"

    echo "<<< Finished ${MODEL_NAME}: dataset=${dataset}, num_views=${num_views}"
done

echo ""
echo "============================================"
echo "All UAV dense n-view pose align runs completed: ${MODEL_NAME}"
echo "============================================"