# UAV-SLAM Benchmarking

This directory contains the private scripts used to reproduce the paper experiments.
There are five experiment groups. Each command below runs every method or experiment
in that group. Run all commands from the GeoFF3D repository root.

## Input data

Every scene uses the following directory layout:

```text
scene_name/
├── images/                 # RGB images
├── cams/                   # camera files with matching image stems
└── depth/                  # metric EXR depth maps with matching image stems
```

The scene lists and dataset roots are configured in the YAML file beside each runner:

| Experiment | Scene list |
| --- | --- |
| Ours | `benchmarking/bash_scripts/ours/default_scenes.yaml` |
| Streaming | `benchmarking/bash_scripts/stream/stream_scenes.yaml` |
| Optimization | `benchmarking/bash_scripts/optim/vggt_slam_scenes.yaml` |
| Ablation | `benchmarking/bash_scripts/ablation/ablation_scenes.yaml` |
| Scalability | `benchmarking/bash_scripts/efficiency_scalability/test_scenes.yaml` |

In each YAML file, set `root` to the dataset directory and list the scene folder names
under `scenes`. Dataset-level or scene-level `params` can be used for settings such as
input stride.

## 1. Our SLRF methods

```bash
python benchmarking/bash_scripts/ours/run_all.py
```

This command runs GeoFF3D, Pi3X, VGGT, and VGGT-Omega through the current SLRF
pipeline on every enabled scene. GeoFF3D is our primary method. It uses the camera
translation prior from `cams/`, estimates spatial footprints, partitions the scene,
reconstructs individual chunks, and aligns them into a global reconstruction.

The default GeoFF3D configuration uses sequential footprint estimation and predicted
depth priors, so `depth/` is not required for this configuration. Using
`footprint_estimation=prior` requires both `cams/` and `depth/`; using
`depth_prior=input` also requires `depth/`. EXR depth values are interpreted in metres.

Run only selected SLRF backbones with:

```bash
python benchmarking/bash_scripts/ours/run_all.py --methods geoff3d
python benchmarking/bash_scripts/ours/run_all.py --methods geoff3d pi3x
```

Useful common options are:

```bash
# Inspect all generated commands without reconstruction.
python benchmarking/bash_scripts/ours/run_all.py --dry-run

# Select the GPU or recompute completed results.
python benchmarking/bash_scripts/ours/run_all.py --cuda-device 1 --overwrite
```

Configure the four checkpoints in `default_scenes.yaml`:

```yaml
methods:
  geoff3d:
    checkpoint: /path/to/geoff3d.pth
  pi3x:
    checkpoint: /path/to/pi3x.pth
  vggt:
    checkpoint: /path/to/vggt.pth
  vggt_omega:
    checkpoint: /path/to/vggt_omega.pth
```

Method-specific environment variables such as `GEOFF3D_CHECKPOINT` remain available
as temporary overrides; configuration files are the recommended persistent interface.

Default output:

```text
experiments/benchmarking/slrf/
├── geoff3d/<dataset>/<scene>/
├── pi3x/<dataset>/<scene>/
├── vggt/<dataset>/<scene>/
└── vggt_omega/<dataset>/<scene>/
```

Each scene output contains `result.rrd`, `result.json`, `processing_time.json`, the
fused point cloud at `eval/pred_points.ply`, and evaluation results at
`eval/metrics.json`. The value passed as `output_path` is a directory, not an RRD
filename.

## 2. Streaming methods

```bash
bash benchmarking/bash_scripts/stream/run_all.sh
```

This command runs LingBot-Map, STream3R, StreamVGGT, and TTT3R on all scenes enabled in
`stream_scenes.yaml`. Every method exports its prediction as RRD and then computes the
common aligned reconstruction metrics. Configure each method's `python` and
`checkpoint` under the top-level `methods` block in `stream_scenes.yaml`.

| Method | Checkpoint/model variable | Default |
| --- | --- | --- |
| LingBot-Map | `LINGBOT_MODEL_PATH` | method default if unset |
| STream3R | `STREAM3R_MODEL_PATH` | `yslan/STream3R` |
| StreamVGGT | `STREAMVGGT_MODEL_PATH` | upstream default/download if unset |
| TTT3R | `TTT3R_MODEL_PATH` | `checkpoints/ttt3r/cut3r_512_dpt_4_64.pth` |

The corresponding environment variables remain optional temporary overrides.

Default output:

```text
experiments/benchmarking/stream/<method>/<dataset>/<scene>.rrd
```

To run only one method, use its individual launcher in the same directory.

## 3. Optimization-based methods

```bash
bash benchmarking/bash_scripts/optim/run_all.sh
```

This command runs VGGT-Long, VGGT-SLAM 2.0, VGGT-SLAM Sim(3), and VGGT-SLAM SL(4) on
all scenes enabled in `vggt_slam_scenes.yaml`, then applies the same aligned evaluation
as the other methods. Select external Python environments with variables such as
`VGGT_SLAM2_PYTHON` and `VGGT_SLAM_PYTHON` when necessary.

Configure `checkpoint` and the VGGT-Long `config` path under `methods` in
`vggt_slam_scenes.yaml`. The default checkpoint is `checkpoints/vggt/model.pt` and the
runner uses the current Python environment. If a separate environment is necessary,
temporarily set `VGGT_SLAM2_PYTHON` or `VGGT_SLAM_PYTHON`.

Default output:

```text
experiments/benchmarking/optim/<method>/<dataset>/<scene>.rrd
```

To run only one method, use its individual launcher in the same directory.

## 4. Ablation experiments

```bash
bash benchmarking/bash_scripts/ablation/run_all.sh
```

This command sequentially runs all SLRF component ablations and the COLMAP dense
baseline configured by `ablation_scenes.yaml`. Completed scenes are skipped; pass
`--overwrite` to recompute them or `--dry-run` to inspect the commands. COLMAP must be
installed separately and can be selected with `COLMAP_BIN=/path/to/colmap`.

Configure both training-stage checkpoints in `ablation_scenes.yaml`:

```yaml
methods:
  geoff3d:
    checkpoints:
      stage2: /path/to/stage2.pth
      stage1: /path/to/stage1.pth
```

All regular ablations select `stage2`; `04_full_stage1` selects `stage1`. A global
`CHECKPOINT` environment variable remains an optional override for every experiment.

Default output:

```text
experiments/benchmarking/ablation/<experiment>/<dataset>/<scene>/
```

After completion, the runner creates `ablation_results.md` and the corresponding JSON
and CSV summaries under `experiments/benchmarking/ablation/`. Detailed definitions of
the individual ablations are in `benchmarking/bash_scripts/ablation/README.md`.

## 5. Efficiency and scalability

```bash
bash benchmarking/bash_scripts/efficiency_scalability/run_all.sh
```

This command benchmarks Pi3X with world-translation priors, VGGT-SLAM 2.0, and
LingBot-Map on the scenes in `test_scenes.yaml`. It records reconstruction/runtime
results and automatically generates the scalability plots. An optional first argument
selects the GPU:

```bash
bash benchmarking/bash_scripts/efficiency_scalability/run_all.sh 1
```

Configure the GeoFF3D, LingBot-Map, and VGGT-SLAM 2.0 checkpoint/Python paths under the
top-level `methods` block in `test_scenes.yaml`.

Default output:

```text
experiments/benchmarking/efficiency_scalability/
```

## Result statistics

Generate Markdown tables from the completed SLRF, streaming, and optimization results:

```bash
python benchmarking/stats.py
```

The script discovers methods, datasets, scenes, and `eval/metrics.json` files from the
current output tree. Select metrics or benchmark groups as needed:

```bash
python benchmarking/stats.py ate_rmse acc_mean fscore_1.0
python benchmarking/stats.py --groups ablation --methods 03_full,04_full_stage1
python benchmarking/stats.py --groups efficiency_scalability
python benchmarking/stats.py --output experiments/benchmarking/results.md
```

Use `--list-methods` and `--list-metrics` to inspect the available inputs.

## Original projects

| Method | Original GitHub project |
| --- | --- |
| GeoFF3D | [yanxian-ll/GeoFF3D](https://github.com/yanxian-ll/GeoFF3D) |
| Pi3 / Pi3X | [yyfz/Pi3](https://github.com/yyfz/Pi3) |
| VGGT | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) |
| VGGT-Omega | [facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega) |
| COLMAP | [colmap/colmap](https://github.com/colmap/colmap) |
| LingBot-Map | [Robbyant/lingbot-map](https://github.com/Robbyant/lingbot-map) |
| STream3R | [NIRVANALAN/STream3R](https://github.com/NIRVANALAN/STream3R) |
| StreamVGGT | [wzzheng/StreamVGGT](https://github.com/wzzheng/StreamVGGT) |
| TTT3R | [Inception3D/TTT3R](https://github.com/Inception3D/TTT3R) |
| VGGT-SLAM / VGGT-SLAM 2.0 | [MIT-SPARK/VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM) |
| VGGT-Long | [DengKaiCQ/VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long) |
