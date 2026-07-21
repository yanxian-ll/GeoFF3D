# UAV Dense N-View：对齐方式评测

本目录用于比较多视图前馈重建在不同全局对齐协议下的结果。这里包含两套不同的评测坐标协议，使用时不能只看 `alignment_mode`，还需要确认脚本调用的是哪个 benchmark 文件。

## 两套评测协议

### `benchmark.py`：view-0 相对坐标协议

大多数脚本调用：

```text
benchmarking/dense_n_view/benchmark.py
```

GT 和预测首先分别转换到各自的第 0 个相机坐标系，然后执行 Sim(3) 对齐。该协议适合比较常规前馈模型的相对多视图重建质量，但不能评价模型是否直接预测到了原始地理世界坐标系。

支持：

```text
pose
points
pose_points
```

不支持 `none`。如果需要无对齐的绝对世界坐标评测，应使用 `benchmark_absolute_world.py`。

### `benchmark_absolute_world.py`：绝对世界坐标协议

当前目录中的 `pi3x_transup_p_woalign.sh` 调用：

```text
benchmarking/dense_n_view/benchmark_absolute_world.py
```

GT 保持数据集世界坐标，预测保持模型输出世界坐标，不转换到 view-0。该协议用于判断 world-translation 模型是否能够直接输出正确的度量尺度、位置和 z-up 坐标系。

支持：

```text
none
pose
points
```

## Alignment modes

### `none`

不对预测执行任何后处理变换：

```text
X_eval = X_pred
```

只有 `benchmark_absolute_world.py` 支持该模式。

适合评估：

- 模型是否直接输出到原始世界坐标系；
- 绝对尺度和绝对位置是否正确；
- world-translation prior 是否真正成为坐标锚点；
- 是否仍然依赖后处理才能恢复世界坐标。

运行：

```bash
ALIGNMENT_MODE=none bash \
  bash_scripts/benchmark/uav_dense_n_view_pose_align/pi3x_transup_p_woalign.sh 0
```

### `pose`

使用预测相机中心与原始 GT 相机中心的对应关系估计一个全局 Sim(3)：

```text
C_gt = s R C_pred + t
```

同一个变换随后应用于预测相机和稠密几何。

优点：

- 不使用 GT 稠密点对应；
- 与常见 ATE/trajectory alignment 协议接近；
- 适合比较不同模型在去除全局坐标系差异后的重建质量。

局限：

- 相机接近共线、基线很短或视图数较少时，Sim(3) 可能退化；
- 会消除模型绝对尺度、朝向和位置方面的误差；
- 不适合单独证明模型能够直接预测绝对坐标。

外部方法默认使用 `pose` Sim(3)。Ours-T/Ours-TR 分别报告 `pose`、`pose_yaw` 和 `none` 三种协议。

示例：

```bash
ALIGNMENT_MODE=pose \
bash bash_scripts/benchmark/uav_dense_n_view_pose_align/pi3x_transup_p.sh 0
```

### `points`

从 GT 与预测 pointmap 的同像素稠密对应中采样点，鲁棒估计全局 Sim(3)：

```text
X_gt = s R X_pred + t
```

同一个变换应用于点图、融合点云和相机位姿。

优点：

- 对相机轨迹退化通常比 pose alignment 更稳定；
- 稠密对应数量多；
- 更关注去除全局 gauge 后的局部几何质量。

局限：

- 对齐本身使用了 GT 稠密几何；
- 可能掩盖预测相机与预测点云之间的全局放置误差；
- 不代表实际部署时能够获得这种对齐信息。

示例：

```bash
ALIGNMENT_MODE=points \
bash bash_scripts/benchmark/uav_dense_n_view_pose_align/pi3x_transup_p.sh 0
```

### `pose_points`

仅 `benchmark.py` 支持。它独立估计两套 Sim(3)：

```text
points Sim(3) → pointmap、depth、融合点云
pose Sim(3)   → camera pose
```

该模式用于分别回答：

- 在最佳稠密几何对齐下，点云质量如何；
- 在最佳相机轨迹对齐下，pose 质量如何。

它不要求同一个 Sim(3) 同时解释相机和点云，因此不适合衡量系统级 camera/geometry 一致性。论文中使用时应明确写成“separately aligned pose and geometry”。

示例：

```bash
ALIGNMENT_MODE=pose_points \
bash bash_scripts/benchmark/uav_dense_n_view_pose_align/pi3x_transup_p.sh 0
```

### `pose_yaw`

从相机中心估计一套统一尺度、绕世界 Z 轴的 yaw 和三维平移，并将同一变换应用于相机和稠密几何，不允许修正 roll/pitch。求解算法与 `predict_scene_to_rrd_spatial.py` 的 `pose_scale_yaw_translation` 完全共用。

```bash
bash bash_scripts/benchmark/uav_dense_n_view_pose_align/pi3x_transup_t_yaw.sh 0
bash bash_scripts/benchmark/uav_dense_n_view_pose_align/pi3x_transup_p_yaw.sh 0
```

## 推荐对比

对于 world-translation 模型，建议至少报告：

| 结果 | Benchmark | Alignment | 说明 |
|---|---|---|---|
| Direct absolute | `benchmark_absolute_world.py` | `none` | 模型直接绝对坐标预测能力 |
| Pose aligned | `benchmark_absolute_world.py` 或 `benchmark.py` | `pose` | 去除全局 pose gauge 后的结果 |
| Dense aligned | `benchmark.py` | `points` | 去除全局几何 gauge 后的局部质量 |

其中 `none` 与 `pose` 的差距反映后处理对齐带来的收益，但要注意 `benchmark.py` 和 `benchmark_absolute_world.py` 的原始坐标协议不同，最严格的直接对比应在 `benchmark_absolute_world.py` 内使用相同输入分别运行 `none` 和 `pose`。

## 脚本说明

- `pi3x_transup_t.sh`：world translation prior，不输入 world rotation prior，默认 `pose`。
- `mapa_t_ft.sh`：finetuned MapAnything，仅输入相对参考视图 0 的 translation vector（`t_i - t_0`），复用原生 `cam_trans_encoder`，不输入 rotation、ray 或 depth prior。
- `pi3x_transup_p.sh`：world translation + world rotation prior，默认 `pose`。
- `pi3x_transup_t_yaw.sh`：translation prior，使用 `pose_yaw`。
- `pi3x_transup_p_yaw.sh`：translation + rotation prior，使用 `pose_yaw`。
- `pi3x_transup_p_woalign.sh`：使用绝对世界坐标 benchmark，并保持默认 `none`。
- `*_ft.sh`：使用对应 finetuned checkpoint。
- `da3*.sh`、`mapa*.sh`、`pi3*.sh`、`vggt*.sh`：统一默认使用 `pose` Sim(3)。

外部方法和 Ours 的标准对齐脚本使用：

```bash
ALIGNMENT_MODE="${ALIGNMENT_MODE:-pose}"
```

Yaw 脚本使用 `pose_yaw`，`*_woalign.sh` 使用 `none`。

## 双卡批量运行

目录中的实验脚本已平均分配到两个串行总控脚本。分别启动后，CUDA 0 和 CUDA 1 并行工作，每张卡内部依次执行任务：

```bash
bash bash_scripts/benchmark/uav_dense_n_view_pose_align/run_cuda0.sh &
bash bash_scripts/benchmark/uav_dense_n_view_pose_align/run_cuda1.sh &
wait
```

任一子实验失败时，对应 GPU 的总控脚本会立即停止，另一张 GPU 不受影响。

## 输出与缓存

Hydra 输出目录由各脚本的 `hydra.run.dir` 决定。不同 alignment mode 应使用不同的 `RUN_TAG` 或输出目录，否则已有 JSON 结果可能被复用或覆盖。

例如：

```bash
RUN_TAG=pi3x_p_pose ALIGNMENT_MODE=pose bash \
  bash_scripts/benchmark/uav_dense_n_view_pose_align/pi3x_transup_p.sh 0

RUN_TAG=pi3x_p_points ALIGNMENT_MODE=points bash \
  bash_scripts/benchmark/uav_dense_n_view_pose_align/pi3x_transup_p.sh 0
```

绝对坐标结果：

```bash
RUN_TAG=pi3x_p_absolute ALIGNMENT_MODE=none bash \
  bash_scripts/benchmark/uav_dense_n_view_pose_align/pi3x_transup_p_woalign.sh 0
```
