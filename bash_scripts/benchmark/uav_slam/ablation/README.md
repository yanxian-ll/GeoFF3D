# UAV-SLAM component ablation

The ablation starts from the adaptive footprint tree with hierarchical Sim(3)
post-alignment. Yaw alignment and cross-chunk prior propagation are then added
in both possible orders:

| Output directory | Partition | Hierarchical post-alignment | Propagated depth prior |
|---|---|---|---|
| `00_temporal_base` | temporal sequence, 25% adjacent overlap | sequential adjacent Sim(3) | off |
| `01_base` | footprint tree | Sim(3) | off |
| `02_yaw_before_propagation` | footprint tree | yaw + translation | off |
| `02_propagation_before_yaw` | footprint tree | Sim(3) | on |
| `03_full` | footprint tree | yaw + translation | on |
| `04_full_stage1` | footprint tree | yaw + translation | on |

The two cumulative paths are:

1. Base -> Yaw -> Full (+ propagation)
2. Base -> Propagation -> Full (+ yaw)

`00_temporal_base` is an isolated temporal baseline. Frames are divided into
chronological windows with the same maximum chunk size as our method and 25\%
overlap between consecutive windows. Chunks retain their sequence order, and
chunk $i$ is aligned only to chunk $i-1$ using their shared-view point clouds.
Both per-chunk camera-pose anchoring and cross-chunk alignment use Sim(3).

For the footprint-tree component experiments, per-chunk prediction-to-pose-prior
alignment (`ALIGN`) is fixed to `pose_scale_yaw_translation`; the `Sim(3)`/`Yaw`
labels in the table refer only to hierarchical post-alignment. The isolated
temporal baseline instead uses `pose_sim3` together with sequential Sim(3)
cross-chunk alignment.

`04_full_stage1` uses the stage-1 `checkpoint-last.pth`; all of its inference
settings are identical to `03_full`, which uses the stage-2
`checkpoint-best.pth`. The generated report includes a separate **Training
stage comparison** table containing only these two runs.

Per-chunk point-cloud logging is enabled for all experiments. The PLY files are
written to
`outputs/ablation/<experiment>/<dataset>/<scene>/chunk_outputs/ply/`.
The ablation also retains `chunk_cache/`: each NPZ contains per-view point maps
and valid masks, while `manifest.json` records global image indices, overlap
sets, adjacency, parent relations, and final post-alignment transforms. This is
the complete information required to recompute every Seam Error edge.

Run all experiments sequentially on GPU 0:

```bash
bash bash_scripts/benchmark/uav_slam/ablation/run_all.sh 0
```

Completed scenes are skipped by default. Pass `--overwrite` to rerun them:

```bash
bash bash_scripts/benchmark/uav_slam/ablation/run_all.sh 0 --overwrite
```

For a fast end-to-end smoke test, use:

```bash
bash bash_scripts/benchmark/uav_slam/ablation/run_fast.sh 0
```

It runs only `03_full`, using roughly 108 frames (`stride=24`), `max_side=224`,
and 2,000 sampled points per Seam Error side. This is enough to exercise
multiple chunks, yaw alignment, prior propagation, retained per-view caches,
and metric aggregation. Results are isolated under `outputs/ablation_fast/`
and must not be used as the paper ablation numbers.

After either runner finishes, `ablation_results.md` is generated under the
selected output root. It contains macro-average reconstruction and seam
metrics, efficiency statistics, and per-scene results. The JSON and CSV Seam
Error summaries are written alongside it.
