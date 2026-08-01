# UAV-SLAM component ablation

The ablation starts from the adaptive footprint tree with hierarchical Sim(3)
post-alignment. Yaw alignment and cross-chunk prior propagation are then added
in both possible orders:

| Output directory | Partition | Hierarchical post-alignment | Propagated depth prior | Model pose input | Pose noise |
|---|---|---|---|---|---|
| `00_temporal_base` | temporal sequence, 25% adjacent overlap | sequential adjacent Sim(3) | off | translation + rotation | mild |
| `01_base` | footprint tree | Sim(3) | off | translation + rotation | mild |
| `02_yaw_before_propagation` | footprint tree | yaw + translation | off | translation + rotation | mild |
| `02_propagation_before_yaw` | footprint tree | Sim(3) | on | translation + rotation | mild |
| `03_full` | footprint tree | yaw + translation | on | translation + rotation | mild |
| `04_full_stage1` | footprint tree | yaw + translation | on | translation + rotation | mild |
| `05_full_stronger_noise` | footprint tree | yaw + translation | on | translation + rotation | stronger |
| `06_full_translation_only` | footprint tree | yaw + translation | on | translation only | mild |
| `07_full_chunk20` | footprint tree, max 20 views/chunk | yaw + translation | on | translation + rotation | mild |
| `08_full_chunk40` | footprint tree, max 40 views/chunk | yaw + translation | on | translation + rotation | mild |
| `09_colmap_dense` | COLMAP sparse mapping + dense MVS | none | off | images only | N/A |

The two cumulative paths are:

1. Base -> Yaw -> Full (+ propagation)
2. Base -> Propagation -> Full (+ yaw)

`00_temporal_base` is an isolated temporal baseline. Frames are divided into
chronological windows with the same maximum chunk size as our method and 25\%
overlap between consecutive windows. Chunks retain their sequence order, and
chunk $i$ is aligned only to chunk $i-1$ using their shared-view point clouds.
Both per-chunk camera-pose anchoring and cross-chunk alignment use Sim(3).

For the footprint-tree component experiments, per-chunk prediction-to-pose-prior
alignment (`align`) is fixed to `scale_yaw_translation`; the `Sim(3)`/`Yaw`
labels in the table refer only to hierarchical post-alignment. The isolated
temporal baseline instead uses `sim3` together with sequential Sim(3)
cross-chunk alignment.

`04_full_stage1` uses the stage-1 `checkpoint-last.pth`; all of its inference
settings are identical to `03_full`, which uses the stage-2
`checkpoint-best.pth`. The generated report includes a separate **Training
stage comparison** table containing only these two runs.

All experiments use perturbed pose metadata by default. The mild setting is
`xy_std=0.5 m`, `z_std=0.8 m`, `yaw_std=1°`, clipped at `2 m`, `2 m`, and
`3°`, respectively. `05_full_stronger_noise` doubles all of those standard
deviations and clipping thresholds while retaining the same seed offset.
`06_full_translation_only` disables the model's input rotation prior with
`ROTATION_PRIOR=none`; all other full-method settings remain unchanged.

All non-chunk-size ablations use a maximum chunk size of 30. The dedicated
chunk-size comparison uses `07_full_chunk20`, `03_full`, and `08_full_chunk40`
for maximum chunk sizes 20, 30, and 40, respectively. Only `MAX_CHUNK_SIZE`
changes; the minimum chunk size remains 8.

`09_colmap_dense` is a classical image-only COLMAP baseline. It runs SIFT
feature extraction, sequential matching, sparse mapping, image undistortion,
PatchMatch stereo, and geometric stereo fusion. It stops reconstruction at
`dense/fused.ply` (no mesh generation), then exports the registered cameras
and dense cloud to RRD/eval format and runs the same aligned pose/point-cloud
metrics as the other methods. It records total reconstruction/export runtime
and peak GPU memory in `processing_time.json`. Seam error is omitted because
COLMAP produces one global reconstruction rather than chunks. COLMAP is an
external dependency and can be selected with `COLMAP_BIN=/path/to/colmap`.
For scalability, sequential matching is the default; set
`COLMAP_MATCHER=exhaustive` to request exhaustive matching.
Because the experiment output may reside on NFS, its SQLite feature database
is built in local scratch (`/tmp/colmap-ablation-$USER` by default) and copied
to the result directory after sparse mapping. Override the local location with
`COLMAP_SCRATCH_ROOT`; it must have enough space for the feature database.
For resolution-controlled comparison, COLMAP also reads locally resized
lossless PNGs throughout feature extraction, SfM, and MVS. The resize exactly
matches the other ablations: longest side 518, then height and width rounded
down to multiples of 14, using `cv2.INTER_AREA`. Each scene records the applied
dimensions in `resize_meta.json`.

Experiments `00` through `08` call the current `scripts/run_slrf.py`; COLMAP
keeps its independent reconstruction/export path. Per-chunk point clouds and
integrated metrics are written to
`experiments/benchmarking/ablation/<experiment>/<dataset>/<scene>/`.

Run all experiments sequentially on GPU 0:

```bash
bash benchmarking/bash_scripts/ablation/run_all.sh --cuda-device 0
```

Completed scenes are skipped by default. Pass `--overwrite` to rerun them:

```bash
bash benchmarking/bash_scripts/ablation/run_all.sh --cuda-device 0 --overwrite
```

To run one experiment, invoke its script with the same options, for example
`03_full.sh --dry-run --cuda-device 0` to inspect every generated command.

After either runner finishes, `ablation_results.md` is generated under the
selected output root. It contains macro-average reconstruction and seam
metrics, efficiency statistics, and per-scene results. The JSON and CSV Seam
Error summaries are written alongside it.
