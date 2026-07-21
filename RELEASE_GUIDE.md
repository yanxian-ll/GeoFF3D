# GeoFF3D 训练与推理发布包

本目录是从原项目整理出的独立源码快照。它包含目标方法的模型、损失、数据加载、两阶段训练、空间 SLAM 推理及评估配置。与目标方法无关的第三方模型源码、数据集、预训练权重、训练 checkpoint 和运行输出不在发布包内。

## 安装

建议使用 Python 3.10 和与 CUDA 匹配的 PyTorch。先安装 PyTorch，再在仓库根目录执行：

```bash
pip install -e .
```

Pi3X 基础权重目录须包含 `config.json` 和 `model.safetensors`。可放在 `checkpoints/pi3x`，也可通过 `PI3X_BASE_MODEL` 指定。

## 数据

训练配置使用仓库原有 MapAnything 数据格式。数据根目录和 metadata 目录可移植配置为：

```bash
export MAPANYTHING_DATA_ROOT=/path/to/data
export MAPANYTHING_METADATA_ROOT=/path/to/data/metadata
export MAPANYTHING_EXPERIMENTS_ROOT=/path/to/experiments
export PI3X_BASE_MODEL=/path/to/pi3x
```

训练集合由以下两个 Hydra 配置定义：

- stage1：`configs/dataset/uavtrain_6d_224_many_ar_16ipg_2g.yaml`
- stage2：`configs/dataset/uavtrain_6d_518_many_ar_16ipg_2g.yaml`

其中引用的各数据集路径模板位于 `configs/dataset/default.yaml`。发布前请确保相应数据和 metadata 已按这些模板准备好。

## 两阶段训练

参数为本机使用的 GPU 数量：

```bash
bash bash_scripts/train/dom/pi3x_zup_translation_finetuning_8v_6d_16ipg_2g.sh 8
bash bash_scripts/train/dom/pi3x_zup_translation_finetuning_8v_6d_16ipg_2g_stage2.sh 8
```

stage2 默认读取 stage1 的 `checkpoint-last.pth`。若 checkpoint 在其他位置：

```bash
export STAGE1_CHECKPOINT=/path/to/stage1/checkpoint-last.pth
bash bash_scripts/train/dom/pi3x_zup_translation_finetuning_8v_6d_16ipg_2g_stage2.sh 8
```

## GNSS 扰动推理

默认使用 `bash_scripts/benchmark/uav_slam/default_scenes.yaml` 中的原始场景列表；也可以准备自己的 scene-list YAML，并通过 `--scene-list` 指定，然后执行：

```bash
export STAGE2_CHECKPOINT=/path/to/stage2/checkpoint-best.pth
bash bash_scripts/benchmark/uav_slam/ours/geoff3d_gnss_perturb.sh \
  --cuda-device 0 \
  --scene-list /path/to/scenes.yaml \
  --overwrite
```

GNSS 噪声可通过原脚本中的环境变量覆盖，例如 `POSE_PERTURB_XY_STD`、`POSE_PERTURB_Z_STD`、`POSE_PERTURB_YAW_STD_DEG`。推理输出默认写入 `outputs/spatial/geoff3d_gnss_perturb/`。

## 不随 GitHub 发布的文件

- 训练/验证数据集；
- `checkpoints/` 下的模型权重；
- `experiments/`、`outputs/` 和缓存；
- 私有路径版 scene-list。

发布 checkpoint 时请单独托管，并在文档中记录其许可证和校验值。
