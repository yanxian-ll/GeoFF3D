#!/bin/bash
export HYDRA_FULL_ERROR=1
export OPENCV_IO_ENABLE_OPENEXR=1

python3 scripts/run_depth_completion.py \
  dataset_root='${root_data_dir}/a3dscenes' \
  batch_size=4 \
  resolution='[1022, 630]' \
  model.pretrained='${root_experiments_dir}/mapanything/training/mapa_finetuning_depth_completion_v3/checkpoint-last.pth' \
  hydra.run.dir='${root_data_dir}/a3dscenes' \
  gpu_id=1
