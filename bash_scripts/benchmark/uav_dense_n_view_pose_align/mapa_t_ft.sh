#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Benchmark finetuned MapAnything with relative-translation-only conditioning:
#   - relative camera translation vector t_i - t_0: enabled
#   - camera rotation prior: disabled
#   - ray/intrinsics prior: disabled
#   - depth prior: disabled

set -euo pipefail

export HYDRA_FULL_ERROR=1
export NUMEXPR_MAX_THREADS=16

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CHECKPOINT_PATH=${CHECKPOINT_PATH:-'${root_experiments_dir}/mapanything/uav_training/mapa_finetuning_16v_6d_16ipg_2g_mvs/checkpoint-best.pth'}

CUDA_DEVICE="${1:-0}"
MODEL_NAME="mapanything"
OUTPUT_NAME="mapa_t_ft"
ALIGNMENT_MODE="${ALIGNMENT_MODE:-pose}"

batch_sizes_and_views=(
    "3 16 benchmark_518_uavff3d_usegeo 16"
)

cd "$ROOT_DIR"

echo "============================================"
echo "UAV Dense N-View Benchmark"
echo "Model:          ${MODEL_NAME}"
echo "Prior:          relative translation vector only"
echo "Alignment mode: ${ALIGNMENT_MODE}"
echo "CUDA device:    ${CUDA_DEVICE}"
echo "============================================"

for combo in "${batch_sizes_and_views[@]}"; do
    read -r batch_size num_views dataset seed <<< "$combo"

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
        model/task=relative_translation_only \
        model.encoder.uses_torch_hub=false \
        model.pretrained="${CHECKPOINT_PATH}" \
        hydra.run.dir='${root_experiments_dir}/pose_align/benchmarking/uav_dense_'"${num_views}"'_view/'"${OUTPUT_NAME}"
done

echo "Completed: ${OUTPUT_NAME}"
