#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONDITION="${CONDITION:?Set CONDITION}"
VALUE="${VALUE:?Set VALUE}"
SEED="${SEED:-16}"
CUDA_DEVICE="${1:-${CUDA_DEVICE:-0}}"
OUTPUT_ROOT="$(realpath -m "${2:-${OUTPUT_ROOT:-$ROOT_DIR/outputs/prior_robustness}}")"
CHECKPOINT_PATH=${PI3X_CHECKPOINT_PATH:-'${root_experiments_dir}/mapanything/uav_training/pi3x_finetuning_16v_6d_16ipg_2g_mvs/checkpoint-best.pth'}

STD_XY=0
STD_Z=0
RETAINED_COUNT=-1
case "$CONDITION" in
  horizontal) STD_XY="$VALUE" ;;
  vertical) STD_Z="$VALUE" ;;
  missing) RETAINED_COUNT="$VALUE" ;;
  *) echo "Unknown CONDITION=$CONDITION" >&2; exit 2 ;;
esac

VALUE_TAG="${VALUE//./p}"
RUN_DIR="$OUTPUT_ROOT/$CONDITION/$VALUE_TAG/seed_${SEED}/pi3x_prior_pose"
mkdir -p "$RUN_DIR"

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 benchmarking/dense_n_view/benchmark_absolute_world.py \
  machine=aws \
  seed="$SEED" \
  compute_abs_metrics=true \
  save_n_fused_ply=0 \
  +alignment_mode=prior_pose \
  +prior_robustness.enabled=true \
  +prior_robustness.std_xy="$STD_XY" \
  +prior_robustness.std_z="$STD_Z" \
  +prior_robustness.missing_ratio=0 \
  +prior_robustness.retained_count="$RETAINED_COUNT" \
  +prior_robustness.input_target=camera_pose \
  +prior_robustness.seed_offset=470001 \
  dataset=benchmark_518_usegeo \
  dataset.principal_point_centered=true \
  dataset.num_workers=12 \
  dataset.num_views=16 \
  batch_size=3 \
  model=pi3x \
  model/task=posed_sfm \
  model.pretrained="$CHECKPOINT_PATH" \
  hydra.run.dir="$RUN_DIR"
