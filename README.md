# GeoFF3D

GeoFF3D is a feed-forward 3D reconstruction framework for UAV imagery. It supports camera, depth, translation, and rotation priors, together with spatial chunking for large scenes.

## Installation

Python 3.10+ and a CUDA-compatible PyTorch installation are recommended.

```bash
git clone https://github.com/yanxian-ll/GeoFF3D.git
cd GeoFF3D
pip install -e .
```

Place pretrained weights under `checkpoints/`. The default Pi3X path is `checkpoints/pi3x`; override it with `PI3X_BASE_MODEL` if needed.

Configure the data and output paths:

```bash
export DATA_ROOT=/path/to/data
export METADATA_ROOT=/path/to/metadata
export EXPERIMENTS_ROOT=/path/to/experiments
```

Dataset configurations are located in `configs/dataset/`.

## Training

The argument is the number of GPUs.

```bash
# GeoFF3D
bash bash_scripts/train/geoff3d_stage1.sh 2
bash bash_scripts/train/geoff3d_stage2.sh 2

# Baselines
bash bash_scripts/train/pi3_finetuning.sh 2
bash bash_scripts/train/pi3x_finetuning.sh 2
bash bash_scripts/train/vggt_finetuning.sh 2
bash bash_scripts/train/vggt_omega.sh 2
```

Stage 2 loads the default Stage 1 output automatically. To use another checkpoint:

```bash
STAGE1_CHECKPOINT=/path/to/checkpoint-last.pth \
bash bash_scripts/train/geoff3d_stage2.sh 2
```

## Large-scene reconstruction

Each launcher reconstructs one scene directly with `scripts/run_slrf.py`:

```bash
bash bash_scripts/run_slrf/geoff3d.sh \
  /path/to/scene \
  /path/to/output
```

Available scripts:

```text
geoff3d.sh
pi3x.sh
vggt.sh
vggt_omega.sh
```

To avoid GT-depth footprints, set
`FOOTPRINT_ESTIMATION=sequential` on any method script. A preliminary
pass sorts the inputs into non-overlapping sequential chunks and merges a tail
smaller than 8 images into the previous chunk. Its aligned predictions provide
the footprints for the normal spatial chunking pass. GeoFF3D uses
`scale_yaw_translation`; Pi3X, VGGT, and VGGT-Omega use `sim3`.

Each script uses its matching checkpoint by default. A custom checkpoint and
additional Hydra overrides can be supplied with:

```bash
CHECKPOINT=/path/to/checkpoint-best.pth \
bash bash_scripts/run_slrf/geoff3d.sh \
  /path/to/scene \
  /path/to/output \
  max_chunk_size=24
```

## Benchmarking

The private reproducibility workspace under `benchmarking/` contains unified runners
for our SLRF methods, streaming and optimization-based baselines, ablations, and
efficiency/scalability experiments. Dataset, checkpoint, and method paths are configured
in the YAML file beside each runner. Results are written to
`experiments/benchmarking/`.

```bash
python benchmarking/bash_scripts/ours/run_all.py
bash benchmarking/bash_scripts/stream/run_all.sh
bash benchmarking/bash_scripts/optim/run_all.sh
bash benchmarking/bash_scripts/ablation/run_all.sh
bash benchmarking/bash_scripts/efficiency_scalability/run_all.sh
```

See [benchmarking/README.md](benchmarking/README.md) for input formats, configuration,
outputs, checkpoints, and result statistics.

## Acknowledgements

This project is developed primarily upon [MapAnything](https://github.com/facebookresearch/map-anything). We sincerely thank its authors for releasing their excellent work. We also thank the authors of [Pi3](https://github.com/yyfz/Pi3), [VGGT](https://github.com/facebookresearch/vggt), and the related open-source projects used by this repository.

## License

See [LICENSE](LICENSE).

## Citation

If this repository is useful to your research, please cite the [UAVFF3D paper](https://arxiv.org/abs/2605.17942):

```bibtex
@article{yang2026uavff3d,
  title={UAVFF3D: A Geometry-Aware Benchmark for Feed-Forward UAV 3D Reconstruction},
  author={Yang, Xiang and Wang, Yongli and Li, HaiFeng and Zhang, Yunsheng},
  journal={arXiv preprint arXiv:2605.17942},
  year={2026}
}
```
