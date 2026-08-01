#!/bin/bash

NUM_GPUS=$1

export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=24
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node ${NUM_GPUS} \
    scripts/train.py \
    machine=aws \
    dataset=uavtrain_4d_224_many_ar_12ipg_2g \
    dataset.num_workers=12 \
    dataset.num_views=12 \
    dataset.min_num_views=2 \
    dataset.principal_point_centered=true \
    dataset.world_augmentation.enabled=true \
    loss=geoff3d_stage1_loss \
    model=geoff3d \
    model.model_config.pretrained_model_name_or_path="${PI3X_BASE_MODEL:-./checkpoints/pi3x}" \
    model.task.world_translation_prior_jitter.enabled=false \
    train_params=geoff3d_stage1 \
    train_params.epochs=80 \
    train_params.warmup_epochs=2 \
    train_params.accum_iter=24 \
    train_params.keep_freq=5 \
    train_params.max_num_of_imgs_per_gpu=12 \
    hydra.run.dir='${root_experiments_dir}/uav_training/geoff3d_stage1_geoff3d_12v_4d_12ipg_2g'
