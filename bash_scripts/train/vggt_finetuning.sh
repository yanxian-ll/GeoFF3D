#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

NUM_GPUS=$1

export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=INFO

export OMP_NUM_THREADS=24
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node ${NUM_GPUS} \
    scripts/train.py \
    machine=aws \
    dataset=uavtrain_4d_518_many_ar_8ipg_2g \
    dataset.num_workers=12 \
    dataset.num_views=8 \
    dataset.principal_point_centered=true \
    loss=vggt_loss \
    model=vggt \
    model.model_config.pretrained_model_name_or_path="./checkpoints/vggt" \
    model.model_config.gradient_checkpointing=true \
    train_params=vggt_finetune \
    train_params.lr=5e-06 \
    train_params.min_lr=5e-08 \
    'train_params.submodule_configs={model.aggregator.patch_embed:{lr:1e-07,min_lr:1e-09}}' \
    train_params.epochs=10 \
    train_params.warmup_epochs=2 \
    train_params.accum_iter=16 \
    train_params.keep_freq=3 \
    train_params.max_num_of_imgs_per_gpu=8 \
    hydra.run.dir='${root_experiments_dir}/uav_training/vggt_finetuning_8v_4d_8ipg_2g'
