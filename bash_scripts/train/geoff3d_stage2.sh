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
    machine=github_release \
    dataset=uavtrain_6d_518_many_ar_16ipg_2g \
    dataset.num_workers=12 \
    dataset.num_views=8 \
    dataset.min_num_views=2 \
    dataset.train.image_transform_mode=per_sample \
    dataset.principal_point_centered=true \
    dataset.world_frame_augmentation.enabled=true \
    dataset.world_frame_augmentation.rotation_deg='[0.0,0.0,180.0]' \
    dataset.world_frame_augmentation.translation_range=10.0 \
    dataset.world_frame_augmentation.scale_range='[0.9,1.2]' \
    dataset.world_frame_augmentation.recenter=true \
    loss=geoff3d_stage2_loss \
    model=geoff3d \
    model/task=world_translation_prior \
    model.task.overall_prob=1.0 \
    model.task.ray_dirs_prob=0.5 \
    model.task.world_translation_prob=0.7 \
    model.task.world_rotation_prob=0.5 \
    model.task.depth_prob=0.5 \
    model.task.sparse_depth_prob=0.5 \
    model.task.sparsification_removal_percent=0.9 \
    model.model_config.pretrained_model_name_or_path="${PI3X_BASE_MODEL:-./checkpoints/pi3x}" \
    model.pretrained="${STAGE1_CHECKPOINT:-${GEOFF3D_EXPERIMENTS_ROOT:-./experiments}/dom/uav_training/geoff3d_stage1/checkpoint-last.pth}" \
    model.model_config.translation_normalization=mean \
    model.model_config.de_normalize_outputs=true \
    model.model_config.force_rotation_prior_for_degenerate_translation=false \
    model.model_config.use_translation_residual_anchor=false \
    model.model_config.translation_residual_anchor_delta_scale=0.1 \
    model.model_config.world_translation_prior_jitter.enabled=true \
    model.model_config.world_translation_prior_jitter.apply_prob=0.75 \
    model.model_config.world_translation_prior_jitter.scale_by_residual_anchor_delta=false \
    model.model_config.world_translation_prior_jitter.std_xy=0.03 \
    model.model_config.world_translation_prior_jitter.std_z=0.10 \
    model.model_config.world_translation_prior_jitter.bias_std_xy=0.01 \
    model.model_config.world_translation_prior_jitter.bias_std_z=0.03 \
    train_params=geoff3d_stage2 \
    train_params.lr=5e-06 \
    train_params.min_lr=5e-08 \
    train_params.epochs=10 \
    train_params.warmup_epochs=2 \
    train_params.accum_iter=16 \
    train_params.keep_freq=3 \
    train_params.max_num_of_imgs_per_gpu=8 \
    train_params.raw_translation_loss_warmup_steps=0 \
    train_params.raw_translation_loss_start_step=0 \
    hydra.run.dir='${root_experiments_dir}/dom/uav_training/geoff3d_stage2'
