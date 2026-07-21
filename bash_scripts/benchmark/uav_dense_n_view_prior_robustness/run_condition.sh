#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CONDITION="${CONDITION:?Set CONDITION to horizontal, vertical, or missing}"
VALUE="${VALUE:?Set VALUE}"
ALIGNMENT_MODE="${ALIGNMENT_MODE:-none}"
SEED="${SEED:-16}"
CUDA_DEVICE="${1:-${CUDA_DEVICE:-0}}"

MODEL_NAME="${MODEL_NAME:-geoff3d}"
CHECKPOINT_PATH=${CHECKPOINT_PATH:-'${root_experiments_dir}/dom/uav_training/pi3x_zup_translation_yaw_aug_8v_6d_16ipg_2g_stage2/checkpoint-best.pth'}
DATASET="${DATASET:-benchmark_518_usegeo}"
NUM_VIEWS="${NUM_VIEWS:-16}"
BATCH_SIZE="${BATCH_SIZE:-3}"
NUM_WORKERS="${NUM_WORKERS:-12}"
OUTPUT_ROOT="$(realpath -m "${2:-${OUTPUT_ROOT:-$ROOT_DIR/outputs/prior_robustness}}")"

STD_XY=0
STD_Z=0
MISSING_RATIO=0
RETAINED_COUNT=-1
case "$CONDITION" in
  horizontal) STD_XY="$VALUE" ;;
  vertical) STD_Z="$VALUE" ;;
  missing) RETAINED_COUNT="$VALUE" ;;
  *) echo "Unknown CONDITION=$CONDITION" >&2; exit 2 ;;
esac

VALUE_TAG="${VALUE//./p}"
RUN_DIR="$OUTPUT_ROOT/$CONDITION/$VALUE_TAG/seed_${SEED}/$ALIGNMENT_MODE"
mkdir -p "$RUN_DIR"

echo "[INFO] Output root: $OUTPUT_ROOT"

if [[ -s "$RUN_DIR/per_dataset_results.json" ]] \
  && grep -q '"protocol_version": 7' "$RUN_DIR/prior_robustness_config.json" 2>/dev/null; then
  echo "[SKIP] Existing result: $RUN_DIR"
  exit 0
fi

cd "$ROOT_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python3 benchmarking/dense_n_view/benchmark_absolute_world.py \
  machine=aws \
  seed="$SEED" \
  compute_abs_metrics=true \
  save_n_fused_ply=0 \
  +alignment_mode="$ALIGNMENT_MODE" \
  +filter_abs_failures=true \
  +max_abs_pose_ate=200.0 \
  +max_abs_pointmap_rmse=200.0 \
  +max_abs_chamfer=200.0 \
  +prior_robustness.enabled=true \
  +prior_robustness.std_xy="$STD_XY" \
  +prior_robustness.std_z="$STD_Z" \
  +prior_robustness.missing_ratio="$MISSING_RATIO" \
  +prior_robustness.retained_count="$RETAINED_COUNT" \
  +prior_robustness.seed_offset=470001 \
  dataset="$DATASET" \
  dataset.principal_point_centered=true \
  dataset.num_workers="$NUM_WORKERS" \
  dataset.num_views="$NUM_VIEWS" \
  batch_size="$BATCH_SIZE" \
  model="$MODEL_NAME" \
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
  model.pretrained="$CHECKPOINT_PATH" \
  hydra.run.dir="$RUN_DIR"
