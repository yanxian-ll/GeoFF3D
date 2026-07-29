#!/bin/bash

NUM_GPUS=$1

export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=24
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Stage 2 uses independent per-view translation-prior dropout because the
# sampled views are not temporally contiguous. Translation jitter models errors.

torchrun --nproc_per_node ${NUM_GPUS} \
    scripts/train.py \
    machine=aws \
    dataset=uavtrain_4d_518_many_ar_8ipg_2g \
    dataset.num_workers=12 \
    dataset.num_views=8 \
    dataset.min_num_views=2 \
    dataset.train.image_transform_mode=per_sample \
    dataset.principal_point_centered=true \
    dataset.world_augmentation.enabled=true \
    loss=geoff3d_stage2_loss \
    model=geoff3d \
    model.task.world_translation_prob=0.7 \
    model.model_config.pretrained_model_name_or_path="${PI3X_BASE_MODEL:-./checkpoints/pi3x}" \
    model.pretrained="${STAGE1_CHECKPOINT:-${GEOFF3D_EXPERIMENTS_ROOT:-./experiments}/uav_training/geoff3d_stage1_geoff3d_12v_4d_12ipg_2g/checkpoint-last.pth}" \
    model.model_config.translation_normalization=mean \
    model.model_config.de_normalize_outputs=true \
    model.model_config.force_rotation_prior_for_degenerate_translation=false \
    model.model_config.world_translation_prior_jitter.enabled=true \
    train_params=geoff3d_stage2 \
    train_params.lr=5e-06 \
    train_params.min_lr=5e-08 \
    train_params.epochs=10 \
    train_params.warmup_epochs=2 \
    train_params.accum_iter=16 \
    train_params.keep_freq=3 \
    train_params.max_num_of_imgs_per_gpu=8 \
    hydra.run.dir='${root_experiments_dir}/uav_training/geoff3d_8v_4d_8ipg_2g'
