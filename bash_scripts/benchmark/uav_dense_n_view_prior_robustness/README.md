# Translation-prior robustness benchmark

This benchmark produces the three panels for horizontal noise, vertical noise,
and missing translation priors. Perturbations are applied only to explicit
model-input `world_translation` values and masks. Dataset poses remain pristine
and are used only for final metric computation.

Ours-TR is evaluated with two protocols: `prior_pose` Sim(3) and constrained
`prior_yaw` (scale + Z-yaw + 3D translation). Alignment is estimated
only from the noisy/retained input priors and corresponding predicted camera
centers. Pristine GT is used only for final metrics.

By default it uses `benchmark_518_usegeo`. The model receives both world
translation and the original world rotation prior. Noise and missing-prior
perturbations affect only translation; rotation remains unchanged.
For the 16-view missing-prior experiment, the sweep retains exactly
`16, 8, 4, 3, 2` translation priors rather than using missing ratios.

The plots also include a finetuned Pi3X-TR baseline using camera-pose priors and
`prior_pose` Sim(3) alignment. At every noise or
retained-prior level, Pi3X-TR receives the same perturbed translations and
availability mask as Ours while pose rotations remain unchanged. Full Sim(3)
results are unavailable with fewer than three retained correspondences.

Run all conditions on CUDA device 0:

```bash
bash bash_scripts/benchmark/uav_dense_n_view_prior_robustness/run_sweep.sh 0
```

Specify a custom output root as the second positional argument:

```bash
bash bash_scripts/benchmark/uav_dense_n_view_prior_robustness/run_sweep.sh \
  0 /path/to/prior_robustness_results
```

The equivalent environment-variable form is:

```bash
OUTPUT_ROOT=/path/to/prior_robustness_results \
  bash bash_scripts/benchmark/uav_dense_n_view_prior_robustness/run_sweep.sh 0
```

Direct condition runs use the same `[cuda_device] [output_root]` convention:

```bash
CONDITION=horizontal VALUE=1 SEED=16 ALIGNMENT_MODE=prior_yaw \
  bash bash_scripts/benchmark/uav_dense_n_view_prior_robustness/run_condition.sh \
  0 /path/to/prior_robustness_results
```

Run one panel:

```bash
bash bash_scripts/benchmark/uav_dense_n_view_prior_robustness/run_horizontal.sh 0
bash bash_scripts/benchmark/uav_dense_n_view_prior_robustness/run_vertical.sh 0
bash bash_scripts/benchmark/uav_dense_n_view_prior_robustness/run_missing.sh 0
```

The default sweep uses seeds 16, 17, and 18. Override it with, for example,
`SEEDS="16"` for a quick run. Existing complete result directories are skipped.

Plot uncertainty bands show the 95% confidence interval of the pooled
scene/seed trials. The CSV also stores the mean, standard deviation, confidence
interval, and number of scene trials.

Outputs are written under `outputs/prior_robustness/`. The combined plot is
saved as PNG, PDF, SVG, and CSV with basename `prior_robustness_abc`.

The plot uses Times New Roman and supports compact-layout controls:

```bash
python3 bash_scripts/benchmark/uav_dense_n_view_prior_robustness/plot_prior_robustness.py \
  --results-root outputs/prior_robustness \
  --output outputs/prior_robustness/prior_robustness_abc \
  --font-scale 1.0 \
  --padding-ratio 0.02 \
  --show-legend
```

`run_sweep.sh` exposes the same settings through `PLOT_FONT_SCALE`,
`PLOT_PADDING_RATIO`, and `PLOT_SHOW_LEGEND` (0 or 1).

The benchmark feature is off by default. Direct Hydra usage must explicitly set
`+prior_robustness.enabled=true`; otherwise the original batch is passed to the
model unchanged.
