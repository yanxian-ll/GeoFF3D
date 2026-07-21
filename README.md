# GeoFF3D

GeoFF3D is a feed-forward 3D reconstruction model for UAV imagery. It conditions reconstruction on optional camera rays, depth, world translation, and world rotation priors, and includes a spatial chunking pipeline for large scenes.

This repository contains only the code required to train GeoFF3D and run GeoFF3D, Pi3X, and VGGT inference baselines. Datasets, pretrained weights, checkpoints, and generated outputs are not included.

## Installation

Use Python 3.10 or newer. Install a CUDA-compatible PyTorch build first, then install GeoFF3D:

```bash
git clone https://github.com/yanxian-ll/GeoFF3D.git
cd GeoFF3D
python -m pip install -e .
```

The base Pi3X weight directory must contain `config.json` and `model.safetensors`. By default it is read from `checkpoints/pi3x`; use `PI3X_BASE_MODEL` to select another location.

## Data configuration

Training uses Hydra dataset definitions under `configs/dataset/`. Configure portable paths with:

```bash
export GEOFF3D_DATA_ROOT=/path/to/data
export GEOFF3D_METADATA_ROOT=/path/to/data/metadata
export GEOFF3D_EXPERIMENTS_ROOT=/path/to/experiments
export PI3X_BASE_MODEL=/path/to/pi3x
```

The two training datasets are configured by:

- `configs/dataset/uavtrain_6d_224_many_ar_16ipg_2g.yaml`
- `configs/dataset/uavtrain_6d_518_many_ar_16ipg_2g.yaml`

## Training

Stage 1, where the argument is the number of local GPUs:

```bash
bash bash_scripts/train/geoff3d_stage1.sh 8
```

Stage 2 reads the Stage 1 checkpoint from the default experiment directory. To specify it explicitly:

```bash
export STAGE1_CHECKPOINT=/path/to/geoff3d_stage1/checkpoint-last.pth
bash bash_scripts/train/geoff3d_stage2.sh 8
```

## Inference

The retained inference entry points are:

```text
bash_scripts/benchmark/uav_slam/ours/geoff3d.sh
bash_scripts/benchmark/uav_slam/ours/geoff3d_gnss_perturb.sh
bash_scripts/benchmark/uav_slam/ours/pi3x.sh
bash_scripts/benchmark/uav_slam/ours/pi3x_gnss_perturb.sh
bash_scripts/benchmark/uav_slam/ours/vggt.sh
```

`pi3x.sh` and `vggt.sh` run their fine-tuned checkpoints. Set `CHECKPOINT` to
the corresponding checkpoint file. `pi3x_gnss_perturb.sh` evaluates the
fine-tuned Pi3X model with configurable GNSS pose noise.

Run GeoFF3D with a trained checkpoint and a scene-list YAML:

```bash
CHECKPOINT=/path/to/geoff3d/checkpoint-best.pth \
bash bash_scripts/benchmark/uav_slam/ours/geoff3d.sh \
  --cuda-device 0 \
  --scene-list /path/to/scenes.yaml \
  --overwrite
```

Run the GNSS perturbation benchmark:

```bash
CHECKPOINT=/path/to/geoff3d/checkpoint-best.pth \
bash bash_scripts/benchmark/uav_slam/ours/geoff3d_gnss_perturb.sh \
  --cuda-device 0 \
  --scene-list /path/to/scenes.yaml \
  --overwrite
```

The scene-list format is documented in `bash_scripts/benchmark/uav_slam/default_scenes.yaml`. Generated results are written below `outputs/` and are ignored by Git.

## Repository layout

```text
geoff3d/                    Python package, models, training, losses, datasets
configs/                    Hydra training and model configurations
scripts/train.py            Distributed training entry point
scripts/predict_scene_to_rrd_spatial.py
geoff3d/spatial_rrd/        Spatial chunking and reconstruction pipeline
bash_scripts/train/         Two-stage GeoFF3D training
bash_scripts/benchmark/uav_slam/ours/
```

## License

See [LICENSE](LICENSE).
