# Generic Feed-forward SLAM 统一任务队列

当前文档是后续开发的**唯一推荐任务队列入口**。旧文档 `docs/slam_design.md` 和 `docs/slam_task_queue.md` 可作为背景资料保留，但实现状态、下一步任务、优先级、可复用开源来源，以本文档为准。

当前路线：**先 VGGT，再 Pi3，再 Pi3X / MapAnything；后端优化优先 GTSAM-first；暂时冻结 GeoFF3D 主线**。

---

# 0. 维护准则

每次实现、修复、测试或重构一个 SLAM 功能后，必须同步更新本文档。

状态标记：

```text
[DONE]      已实现并通过对应测试或人工验证；
[PARTIAL]   已有代码，但功能不完整、未接真实模型、未覆盖关键测试；
[TODO]      尚未实现；
[BLOCKED]   被依赖项阻塞，暂时不能继续；
[DEFERRED]  有意延后，不是当前优先级；
[FROZEN]    暂时冻结，不作为当前路线推进对象。
```

完成任务必须记录：完成日期、修改文件、测试命令、测试结果、参考/复用来源、是否直接复用外部代码、无法在当前环境完成的测试、遗留问题、下一步。

如果测试依赖大模型、GPU、外部 checkpoint、GTSAM 或私有数据，必须写明原因，并在后续真实环境补测。

实现时必须标注开源参考/复用来源。如果直接复用或显著改写外部代码，必须在文件顶部或函数注释中写明来源 URL、原始文件路径、license。禁止无来源复制外部代码；禁止把外部项目的模型专用逻辑写死到 Generic SLAM 主流程。

---

# 1. 开源项目参考与复用索引

```text
本项目已有代码：
- generate-dom / mapanything/models：模型初始化、external wrapper、输出字段约定。
- slam/models/view_builder.py：SlamFrame → MapAnything-style view dict。
- slam/io/folder_dataset.py：image folder / priors.json / cams_dir → SlamFrame。
- scripts/predict_scene_to_rrd.py：单场景推理、RRD/可视化输出。
- slam/backend/base_backend.py：通用后端接口。

外部开源项目：
- facebookresearch/vggt：VGGT 模型加载、图像预处理、输出字段、推理脚本。对应 T102-T105。
- facebookresearch/map-anything：多模型统一接口、多模态 prior 输入、metric/local 输出。对应 T300-T304, T400-T404。
- rmurai0610/MASt3R-SLAM：tracking、pointmap matching、local fusion、loop closure；后端为 lietorch + 自定义 Gauss-Newton/CUDA。对应 T501, T700-T702。
- MIT-SPARK/VGGT-SLAM@version1.0：VGGT-SLAM 1.0；GTSAM SL4 graph；use_sim3 分支为 scale 外估计 + GTSAM Pose3 graph。关键文件：vggt_slam/graph.py, graph_se3.py, solver.py。对应 T500, T501, T503。
- MIT-SPARK/VGGT-SLAM@main：VGGT-SLAM 2.0；GTSAM SL4 / PriorFactorSL4 / BetweenFactorSL4 / LevenbergMarquardtOptimizer。关键文件：vggt_slam/graph.py, solver.py。对应 T503。
- GREAT-WHU/MASt3R-Fusion：modified GTSAM + gtsam_unstable + visual Sim3 constraints + IMU/GNSS factors。关键文件：mast3r_fusion/global_opt.py。对应 T500, T501, T502。
- ORB-SLAM3 / GTSAM / COLMAP / OpenCV / Open3D：keyframe graph、factor graph、camera model、PnP/RANSAC、PLY/point cloud IO、debug visualization。对应 T500-T503, T600, T700-T702。
- SplaTAM / ARTDECO / M3：Gaussian map、differentiable rendering、camera pose + Gaussian map joint optimization。对应 T602。
```

---

# 2. 当前代码实现快照

## 2.1 [DONE] 基础框架

```text
slam/core/data_types.py
slam/models/model_spec.py
slam/priors/prior_types.py
slam/models/base_adapter.py
slam/models/dummy_adapter.py
slam/core/registry.py
slam/frontend/chunk_manager.py
slam/core/generic_slam.py
slam/priors/prior_builder.py
slam/geometry/coordinate_normalizer.py
slam/geometry/se3.py
slam/geometry/sim3.py
slam/geometry/umeyama.py
slam/frontend/overlap_manager.py
slam/mapping/pointcloud_map.py
slam/mapping/depth_cache.py
scripts/slam/run_generic_slam.py
configs/slam/generic_dummy.yaml
```

## 2.2 [DONE] Folder 输入与调试输出

```text
slam/io/folder_dataset.py
scripts/slam/run_slam_folder.py
tests/slam/test_run_slam_folder_script.py
```

- 已完成：在 slam/io/folder_dataset.py load_folder_frames 函数中，已有 priors_path（json）、cams_dir（txt），已新增 depth_path（exr/npy）的可选输入先验，以及 mask_path（png/npy）的可选 depth_mask 先验；mask 只有在输入 depth 时才生效，单独输入无效；如果只输入 depth，没有输入 mask，默认 depth > 0 为有效深度；已同步修改 scripts/slam/run_slam_folder.py 输入部分和 configs/slam/generic_dummy.yaml。

- 已完成：scripts/slam/run_slam_folder.py 中 argparser 输入参数会在导入 config 后更新到 cfg，后续 load_folder_frames、frontend、backend、export 统一读取 resolved cfg。

- 已完成：slam/core/registry.py 中已添加 VGGT-Omega，并新增轻量 VGGTOmegaAdapter 默认使用 model_name=vggt_omega、data_norm_type=identity。

- 已完成：scripts/slam/run_slam_folder.py 中已添加必要阶段注释，覆盖 load_config、load_folder_frames、adapter、frontend/backend/mapping、run、export；slam/core/generic_slam.py 中空注释已替换为实际流程说明。

- 已完成：CoordinateNormalizer 已改为 config-driven alignment。默认 `alignment.mode=auto`：非 world 输出的第一 chunk 以 identity 作为全局坐标系；后续 chunk 优先按 world prior（若足够）对齐，否则使用 overlap；overlap 默认先尝试重叠帧点云 Sim3（`prefer_overlap_points=true`、`min_overlap_points=100`），点数不足再退回 overlap pose anchor。存在 world_translation 时默认 world prior 优先；如需“先 world 再 overlap 点云精修”，可显式设置 `alignment.refine_world_with_overlap=true`。存在 world_rotation 时可设置 `alignment.use_world_rotation=true` 启用单帧 SE3 world pose anchor。


## 2.3 [DONE] VGGT / Pi3 / Pi3X / MapAnything mock/CI 覆盖

```text
configs/slam/vggt_folder.yaml
tests/slam/test_vggt_folder_config.py
slam/models/vggt_adapter.py
tests/slam/test_vggt_adapter.py
scripts/slam/debug_vggt_single_chunk.py
tests/slam/test_debug_vggt_single_chunk_script.py
tests/slam/test_vggt_multichunk_alignment.py

configs/slam/pi3_folder.yaml
tests/slam/test_pi3_folder_config.py
slam/models/pi3_adapter.py
tests/slam/test_pi3_adapter.py
scripts/slam/debug_pi3_single_chunk.py
tests/slam/test_debug_pi3_single_chunk_script.py
tests/slam/test_pi3_multichunk_alignment.py

configs/slam/pi3x_folder.yaml
tests/slam/test_pi3x_folder_config.py
slam/models/pi3x_adapter.py
tests/slam/test_pi3x_adapter.py
tests/slam/test_pi3x_multichunk_priors.py

configs/slam/mapanything_folder.yaml
tests/slam/test_mapanything_folder_config.py
slam/models/mapanything_adapter.py
tests/slam/test_mapanything_adapter.py
tests/slam/test_mapanything_multichunk_priors.py
tests/slam/test_real_image_local_weights.py
```

2026-06-01 已补充真实 image folder + 本地 checkpoint + RRD opt-in pytest，覆盖 VGGT / Pi3 / Pi3X / MapAnything 30 帧单 chunk 端到端导出。

## 2.4 [PARTIAL] 后端优化

```text
slam/backend/se3_pose_graph.py              # PARTIAL: GTSAM-first Pose3 backend 已实现；真实 VGGT 30 帧多 chunk fallback smoke 已通过；GTSAM 环境测试待补测
slam/backend/sim3_pose_graph.py             # PARTIAL: chunk-level Sim3 anchor backend 已实现；真实 Pi3 30 帧多 chunk overlap 已通过；更长序列待补测
slam/backend/sensor_fusion_graph.py         # PARTIAL: GTSAM-first sensor fusion shell 已实现；真实 VGGT 30 帧多 chunk smoke 已通过；raw IMU preintegration 待实现
slam/backend/sl4_submap_graph.py            # PARTIAL: GTSAM SL4 submap backend 已实现；真实 VGGT 30 帧多 chunk fallback smoke 已通过；gtsam-develop / SL4 环境待补测
slam/mapping/dem_map.py                     # PARTIAL: placeholder
slam/loop/                                  # PARTIAL: baseline retrieval / pose verifier / loop factor adapter 已有测试；真实 dense/geometric loop 待实现
```

## 2.5 [PARTIAL] 输出与导出

```text
slam/io/export_outputs.py                   # PARTIAL: summary/manifest/npz/csv/tum/json/npy/ply/rrd/COLMAP text/chunk prediction debug 导出已实现；真实 30 帧模型已补测；外部 viewer/tool 读取待补测
scripts/slam/run_slam_folder.py             # DONE: 已接统一 exporter
```

---

# 3. 当前推荐执行路线

```text
R1. 在 gtsam-develop 环境验证 Pose3 与 SL4 分支；
R2. 用 evo/Open3D/COLMAP/Rerun viewer 对 T600 导出格式做真实工具链验证；
R3. 增加更长真实序列 / 更大 chunk / sensor 数据组合稳定性测试；
R4. 实现 loop closure、DEM/Gaussian map；
R5. 用 COLMAP/OpenMVS/Open3D/evo/Rerun viewer 人工或工具链读取真实导出文件。
```

---

# 4. 统一任务队列

## P0. 文档与状态维护

### T000 — [DONE] 合并旧任务队列与设计文档
完成日期：2026-05-31
修改文件：`docs/slam_unified_task_queue.md`
测试命令：未运行；文档任务。

### T001 — [DONE] 在任务队列中加入开源参考/复用来源规则
完成日期：2026-05-31
修改文件：`docs/slam_unified_task_queue.md`
测试命令：未运行；文档任务。

### T002 — [TODO] 后续每次功能实现后更新任务状态和来源字段
通过标准：任意功能 PR/commit 都能在本文档中找到对应状态更新。

### T003 — [DONE] 新增真实图像 + 本地权重 opt-in 测试
完成日期：2026-06-01
修改文件：`tests/slam/test_real_image_local_weights.py`, `configs/slam/pi3x_folder.yaml`, `slam/models/pi3x_adapter.py`, `slam/models/mapanything_adapter.py`, `tests/slam/test_mapanything_adapter.py`, `tests/slam/test_real_wrapper_output_standardization.py`, `docs/slam_unified_task_queue.md`
完成内容：新增默认跳过的真实环境 pytest，代码中保留真实 image/cams/checkpoint 路径；每个真实 case 保存 RRD、summary、manifest、npz、ply、csv、tum、cameras.json。真实测试固定 resize 为 `504x896`，避免 DINO/VGGT/Pi3 patch size 14 整除失败。Pi3X 使用真实 wrapper 要求的 `data_norm_type=identity`；MapAnything infer 前移除 `camera_pose/camera_intrinsics/depthmap` 训练别名，仅保留真实 inference 接受的 key。
真实路径：`/opt/data/private/dataset/data/NPU_Dronemap/gopro-npu-kfs/images`, `/opt/data/private/dataset/data/NPU_Dronemap/gopro-npu-kfs/cams`, `checkpoints/vggt/model.pt`, `checkpoints/pi3/model.safetensors`, `checkpoints/pi3x/model.safetensors`, `checkpoints/map-anything-v1/map-anything-v1.pth`
输出目录：`outputs/slam/real_image_local_weights/`
测试命令：`python3 -m pytest tests/slam/test_pi3x_adapter.py tests/slam/test_pi3x_folder_config.py tests/slam/test_mapanything_adapter.py tests/slam/test_real_wrapper_output_standardization.py tests/slam/test_real_image_local_weights.py -q`
测试结果：`16 skipped`（真实测试文件默认跳过）；相关 SLAM 全量测试：`93 passed, 19 skipped in 0.92s`
真实环境测试命令：`source /opt/conda/etc/profile.d/conda.sh && conda activate mapanything && SLAM_RUN_REAL_IMAGE_TESTS=1 python -m pytest tests/slam/test_real_image_local_weights.py -q -s`
真实环境测试结果：`12 passed in 659.26s`
参考/复用来源：`scripts/train.py` 的 `configure_torch_hub` 本地 torch hub 方式；`scripts/predict_scene_to_rrd.py` 的 RRD 保存/人工查看工作流。
是否直接复用外部代码：否。
遗留问题：尚未人工打开 Rerun viewer 检查可视化观感；尚未做更长序列/更大 chunk/sensor 数据组合压力测试。

### T004 — [DONE] 收敛 SLAM 全量单元测试兼容性
完成日期：2026-06-01
修改文件：`slam/core/data_types.py`, `slam/loop/dino_salad_provider.py`, `slam/loop/retrieval.py`, `slam/loop/factor_adapter.py`, `slam/backend/se3_pose_graph.py`, `tests/slam/test_pi3x_multichunk_priors.py`, `docs/slam_unified_task_queue.md`
完成内容：`WorldChunkPrediction` 兼容历史测试中的 `alignment_transform/source_prediction`；DINO/SALAD provider 在无 torch 且注入 lightweight model 时提供 numpy descriptor fallback；EmbeddingLoopRetrieval 过滤 GenericLoopClosureManager 传入的非 provider kwargs；Sim3 loop factor adapter 在无 chunk id 时仍能生成 frame-level overlap prior；SE3 fallback/GTSAM overlap pose 同时接受 `pose` 和 `T_world_cam`；补齐 Pi3X mock 的 pose-prior 记录。
测试命令：`python3 -m pytest tests/slam -q`
测试结果：`93 passed, 19 skipped in 0.92s`
参考/复用来源：本项目现有 loop/backend 测试和数据结构约定；无外部代码复制。
遗留问题：DINO/SALAD 真实 checkpoint 检索、dense pointmap loop verification、真 GTSAM loop factor 长序列验证仍待补测。

### T005 — [DONE] 真实多 chunk / overlap 本地权重端到端测试
完成日期：2026-06-01
修改文件：`tests/slam/test_real_image_local_weights.py`, `slam/io/export_outputs.py`, `tests/slam/test_export_outputs.py`, `tests/slam/test_run_slam_folder_script.py`, `docs/slam_unified_task_queue.md`
完成内容：在真实图像/本地权重 opt-in pytest 中新增 VGGT / Pi3 / Pi3X / MapAnything 的 30 帧、`chunk_size=8`、`overlap=2` 端到端 case；导出 summary/manifest 时记录 `world_prediction_chunks`，测试断言每个 multichunk case 至少 2 个 chunk，实际 4 个模型均为 5 个 world prediction chunks。
真实输出目录：
`outputs/slam/real_image_local_weights/vggt_30f_chunk8_overlap2_no_opt`,
`outputs/slam/real_image_local_weights/pi3_30f_chunk8_overlap2_no_opt`,
`outputs/slam/real_image_local_weights/pi3x_30f_chunk8_overlap2_no_opt`,
`outputs/slam/real_image_local_weights/mapanything_v1_30f_chunk8_overlap2_no_opt`
测试命令：`python3 -m pytest tests/slam -q`
测试结果：`93 passed, 19 skipped in 0.92s`
真实环境测试命令：`source /opt/conda/etc/profile.d/conda.sh && conda activate mapanything && SLAM_RUN_REAL_IMAGE_TESTS=1 python -m pytest tests/slam/test_real_image_local_weights.py -q -s`
真实环境测试结果：`12 passed in 659.26s`
参考/复用来源：本项目现有 chunk/overlap pipeline、`scripts/predict_scene_to_rrd.py` RRD 输出工作流。
是否直接复用外部代码：否。
遗留问题：尚未做更长序列、不同 overlap/chunk 组合、sensor 数据组合压力测试。

### T006 — [DONE] 真实多 chunk / overlap + 后端优化组合测试
完成日期：2026-06-01
修改文件：`tests/slam/test_real_image_local_weights.py`, `docs/slam_unified_task_queue.md`
完成内容：新增真实 30 帧、`chunk_size=8`、`overlap=2` 的 backend multichunk opt-in case，覆盖 VGGT+SE3PoseGraph、Pi3+Sim3PoseGraph、VGGT+SensorFusionGraph、VGGT+SL4SubmapGraph；测试断言完整 T600 导出、RRD/COLMAP text、`world_prediction_chunks >= 2`、backend diagnostics 非空。
真实输出目录：
`outputs/slam/real_image_local_weights/vggt_se3_pose_graph_30f_chunk8_overlap2`,
`outputs/slam/real_image_local_weights/pi3_sim3_pose_graph_30f_chunk8_overlap2`,
`outputs/slam/real_image_local_weights/vggt_sensor_fusion_graph_30f_chunk8_overlap2`,
`outputs/slam/real_image_local_weights/vggt_sl4_submap_graph_30f_chunk8_overlap2`
测试命令：`python3 -m pytest tests/slam -q`
测试结果：`93 passed, 19 skipped in 0.92s`
真实环境测试命令：`source /opt/conda/etc/profile.d/conda.sh && conda activate mapanything && SLAM_RUN_REAL_IMAGE_TESTS=1 python -m pytest tests/slam/test_real_image_local_weights.py -q -s -k multichunk_backends`
真实环境测试结果：`4 passed, 12 deselected in 294.67s`
参考/复用来源：本项目现有 GTSAM-first/fallback backend、chunk overlap pipeline、`scripts/predict_scene_to_rrd.py` RRD 输出工作流。
是否直接复用外部代码：否。
遗留问题：当前环境无 gtsam，因此 SE3/SL4 真实 GTSAM 分支仍待 gtsam-develop 环境补测；SensorFusionGraph 未接 raw sensor 数据。

### T007 — [DONE] 完善 2.2 Folder 输入、脚本 cfg 覆盖与 VGGT-Omega registry
完成日期：2026-06-03
修改文件：`slam/io/folder_dataset.py`, `scripts/slam/run_slam_folder.py`, `configs/slam/generic_dummy.yaml`, `slam/core/registry.py`, `slam/models/vggt_omega_adapter.py`, `slam/models/view_builder.py`, `slam/core/generic_slam.py`, `slam/frontend/overlap_manager.py`, `slam/geometry/coordinate_normalizer.py`, `tests/slam/test_folder_dataset.py`, `tests/slam/test_adapter_registry.py`, `tests/slam/test_overlap_manager.py`, `docs/slam_unified_task_queue.md`
完成内容：`load_folder_frames` 新增 `depth_path` 与 `mask_path`，支持 depth 文件/目录按 frame stem 匹配；mask 仅在 depth 存在时生效；无 mask 时默认 `depth > 0` 有效。`run_slam_folder.py` 改为 load config 后统一应用 CLI 覆盖，后续阶段读取 resolved cfg；`generic_dummy.yaml` 补齐 input/export/alignment 字段。新增 `VGGTOmegaAdapter` 并注册 `vggt_omega`。补齐脚本和主 pipeline 阶段注释。`CoordinateNormalizer` 改为 config-driven alignment，支持 identity/world_priors/overlap/auto 模式、world rotation SE3 anchor、dense overlap point Sim3，以及显式 `refine_world_with_overlap`。
测试命令：`python3 -m pytest tests/slam -q`
测试结果：`97 passed, 19 skipped in 0.92s`
参考/复用来源：本项目已有 `scripts/predict_scene_to_rrd.py` / `scripts/visualize_dataset_rerun.py` 的 EXR/OpenCV 读取思路；本项目已有 MapAnything wrapper 初始化与 adapter 输出标准化约定；MIT-SPARK/VGGT-SLAM `vggt_slam/solver.py` 的 first-submap anchor、overlap submap connection、point-cloud scale estimation 思路，以及 `vggt_slam/graph.py` 的 SL4 factor graph 组织方式。
是否直接复用外部代码：否。
遗留问题：当前轻量测试环境没有 Pillow/OpenCV，因此 PNG/EXR 实际 IO 路径未在本地单测中写读闭环；真实环境需用 `depth/*.exr` 与 `mask/*.png` 补测。`refine_world_with_overlap=true` 的真实精修效果仍需长序列补测。

### T008 — [DONE] 真实测试脚本支持人工直接传入图像与 checkpoints
完成日期：2026-06-03
修改文件：`tests/slam/conftest.py`, `tests/slam/_manual.py`, `tests/slam/_bootstrap.py`, `tests/slam/sitecustomize.py`, `tests/slam/test_real_image_local_weights.py`, `scripts/slam/run_slam_folder.py`, `scripts/slam/debug_vggt_single_chunk.py`, `scripts/slam/debug_pi3_single_chunk.py`, `slam/models/vggt_adapter.py`, `slam/models/pi3_adapter.py`, `tests/slam/test_run_slam_folder_script.py`, 所有 `tests/slam/test_*.py`, `docs/slam_unified_task_queue.md`
完成内容：新增 pytest 参数 `--slam-run-real`, `--slam-image-dir`, `--slam-cams-dir`, `--slam-output-root`, `--slam-resize`, `--slam-vggt-checkpoint`, `--slam-pi3-checkpoint`, `--slam-pi3x-checkpoint`, `--slam-mapanything-checkpoint`；`tests/slam/test_real_image_local_weights.py` 支持直接 `python` 调用并传入真实图像、cams、输出目录和四类 checkpoint。其余所有 `tests/slam/test_*.py` 都已接入统一 direct-run 入口，`python tests/slam/test_x.py -q` 会真正执行该文件测试。`run_slam_folder.py` 新增 `--model_checkpoint`，会按模型类型转换为 Hydra override；VGGT/Pi3 adapter 改为消费 `hydra_overrides`。VGGT/Pi3 单 chunk debug 脚本也支持 `--model_checkpoint`。
手动调用示例：`python tests/slam/test_real_image_local_weights.py --image_dir /path/to/images --cams_dir /path/to/cams --vggt_checkpoint checkpoints/vggt/model.pt --pi3_checkpoint checkpoints/pi3/model.safetensors --pi3x_checkpoint checkpoints/pi3x/model.safetensors --mapanything_checkpoint checkpoints/map-anything-v1/map-anything-v1.pth`
测试命令：`python3 tests/slam/test_real_image_local_weights.py --help`; `python3 -m pytest tests/slam/test_real_image_local_weights.py --help | rg "slam-(run-real|image-dir|vggt-checkpoint|mapanything-checkpoint)"`; 批量直接执行全部 53 个 `tests/slam/test_*.py`（真实测试用 `--help`）；`python3 -m pytest tests/slam -q`
测试结果：手动 help / pytest option help 通过；53 个测试脚本直接调用 `direct_failures=0`；全量 `113 passed, 19 skipped in 1.58s`
参考/复用来源：本项目已有 opt-in 真实测试、`scripts/slam/run_slam_folder.py` 统一入口、Hydra override 模型初始化路径。
是否直接复用外部代码：否。
遗留问题：真实大模型/GPU 跑测未在当前轻量环境执行；需在真实环境用上述命令补测。

### T009 — [DONE] run_slam_folder 支持配置化 loop closure 调试入口
完成日期：2026-06-03
修改文件：`scripts/slam/run_slam_folder.py`, `configs/slam/generic_dummy.yaml`, `slam/backend/no_opt_backend.py`, `tests/slam/test_run_slam_folder_script.py`, `tests/slam/test_pointcloud_registration_verifier.py`, `tests/slam/test_pnp_ransac_verifier.py`, `docs/slam_unified_task_queue.md`
完成内容：`run_slam_folder.py` 新增 `--enable_loop_closure`, `--loop_verifier`, `--loop_distance_threshold`, `--loop_min_temporal_gap`, `--loop_max_translation_error`，并补充 point-cloud registration verifier 的 `--loop_max_rmse`, `--loop_max_correspondence_distance`, `--loop_min_inliers`, `--loop_min_inlier_ratio`, `--loop_transform_type` 参数，以及 PnP/RANSAC verifier 的 `--loop_max_reprojection_error`, `--loop_min_correspondences`, `--loop_ransac_iterations` 参数，可从脚本直接打开 T700-T702 已有 loop retrieval / verifier / factor adapter。`generic_dummy.yaml` 新增 `loop_closure` 默认配置。`NoOptBackend` 可记录 loop factors，便于不启用优化后端时调试 loop pipeline。
测试命令：`python3 -m pytest tests/slam/test_run_slam_folder_script.py tests/slam/test_generic_loop_closure.py tests/slam/test_model_pair_reprojection_verifier.py tests/slam/test_pointcloud_registration_verifier.py tests/slam/test_pnp_ransac_verifier.py tests/slam/test_no_opt_backend.py -q`; `python3 -m pytest tests/slam -q`
测试结果：相关测试通过；全量 `113 passed, 19 skipped in 1.58s`
参考/复用来源：本项目已有 `GenericLoopClosureManager`, `TemporalDistanceLoopRetrieval`, `PoseDistanceVerifier`, `ModelPairReprojectionVerifier`, `PointCloudRegistrationVerifier`, `PnPRansacVerifier`, `LoopFactorAdapter`。
是否直接复用外部代码：否。
遗留问题：真实 long-sequence loop closure、dense matching loop factor、真 GTSAM loop optimization 仍待补测。

---

## P1. VGGT 优先路线

### T100 — [DONE] 实现真实 image folder 输入脚本
修改文件：`scripts/slam/run_slam_folder.py`, `slam/io/folder_dataset.py`, `tests/slam/test_run_slam_folder_script.py`
测试命令：`python -m pytest tests/slam/test_run_slam_folder_script.py -q`

### T101 — [DONE] 新增 VGGT folder 配置
修改文件：`configs/slam/vggt_folder.yaml`, `tests/slam/test_vggt_folder_config.py`
测试命令：`python -m pytest tests/slam/test_vggt_folder_config.py -q`

### T102 — [DONE] 修正并验证 VGGTAdapter 真实模型调用
完成日期：2026-06-01
修改文件：`slam/models/vggt_adapter.py`, `tests/slam/test_vggt_adapter.py`, `tests/slam/test_real_image_local_weights.py`, `docs/slam_unified_task_queue.md`
测试命令：`python -m pytest tests/slam/test_vggt_adapter.py -q`
真实环境测试命令：`source /opt/conda/etc/profile.d/conda.sh && conda activate mapanything && SLAM_RUN_REAL_IMAGE_TESTS=1 python -m pytest tests/slam/test_real_image_local_weights.py -q -s`
真实环境测试结果：`vggt-30-one-chunk-rrd` 和 `vggt-30-overlap` 通过；输出 `outputs/slam/real_image_local_weights/vggt_30f_chunk30_no_opt/vggt_30f_chunk30_no_opt.rrd` 与 `outputs/slam/real_image_local_weights/vggt_30f_chunk8_overlap2_no_opt/vggt_30f_chunk8_overlap2_no_opt.rrd`。
遗留问题：尚未做 VGGT 更长序列/backend 组合真实评估。

### T103 — [DONE] 新增 VGGT 单 chunk smoke 脚本/测试
修改文件：`scripts/slam/debug_vggt_single_chunk.py`, `tests/slam/test_debug_vggt_single_chunk_script.py`
测试命令：`python -m pytest tests/slam/test_debug_vggt_single_chunk_script.py -q`

### T104 — [DONE] VGGT 多 chunk 无 prior overlap 路线测试
修改文件：`tests/slam/test_vggt_multichunk_alignment.py`
测试命令：`python -m pytest tests/slam/test_vggt_multichunk_alignment.py -q`

### T105 — [DONE] VGGT + optional world_translation Sim3 对齐测试
修改文件：`tests/slam/test_vggt_multichunk_alignment.py`
测试命令：`python -m pytest tests/slam/test_vggt_multichunk_alignment.py -q`

---

## P2. Pi3 路线

### T200 — [DONE] 细化 Pi3Adapter ModelSpec
修改文件：`slam/models/pi3_adapter.py`, `tests/slam/test_pi3_adapter.py`
测试命令：`python -m pytest tests/slam/test_pi3_adapter.py -q`

### T201 — [DONE] 实现 Pi3Adapter 输入构造与真实推理
完成日期：2026-06-01
修改文件：`slam/models/pi3_adapter.py`, `tests/slam/test_pi3_adapter.py`, `tests/slam/test_real_image_local_weights.py`, `docs/slam_unified_task_queue.md`
测试命令：`python -m pytest tests/slam/test_pi3_adapter.py -q`
真实环境测试命令：`source /opt/conda/etc/profile.d/conda.sh && conda activate mapanything && SLAM_RUN_REAL_IMAGE_TESTS=1 python -m pytest tests/slam/test_real_image_local_weights.py -q -s`
真实环境测试结果：`pi3-30-one-chunk-rrd` 和 `pi3-30-overlap` 通过；输出 `outputs/slam/real_image_local_weights/pi3_30f_chunk30_no_opt/pi3_30f_chunk30_no_opt.rrd` 与 `outputs/slam/real_image_local_weights/pi3_30f_chunk8_overlap2_no_opt/pi3_30f_chunk8_overlap2_no_opt.rrd`。
遗留问题：尚未做 Pi3 更长序列/backend 组合真实评估。

### T202 — [DONE] 新增 Pi3 folder 配置
修改文件：`configs/slam/pi3_folder.yaml`, `tests/slam/test_pi3_folder_config.py`
测试命令：`python -m pytest tests/slam/test_pi3_folder_config.py -q`

### T203 — [DONE] Pi3 单 chunk smoke test
修改文件：`scripts/slam/debug_pi3_single_chunk.py`, `tests/slam/test_debug_pi3_single_chunk_script.py`
测试命令：`python -m pytest tests/slam/test_debug_pi3_single_chunk_script.py -q`

### T204 — [DONE] Pi3 多 chunk overlap 测试
修改文件：`tests/slam/test_pi3_multichunk_alignment.py`
测试命令：`python -m pytest tests/slam/test_pi3_multichunk_alignment.py -q`

---

## P3. Pi3X 带先验 metric_local 路线

### T300 — [DONE] 细化 Pi3XAdapter ModelSpec
完成日期：2026-05-31
修改文件：`slam/models/pi3x_adapter.py`, `tests/slam/test_pi3x_adapter.py`, `docs/slam_unified_task_queue.md`
完成内容：Pi3XAdapter 明确为 geometry-prior-capable metric_local/metric 模型；支持 intrinsics/rays/pose/world_translation/world_rotation/depth priors；输出 camera/depth/points/confidence；不宣称一定预测 intrinsics。
测试命令：`python -m pytest tests/slam/test_pi3x_adapter.py -q`

### T301 — [DONE] 实现 Pi3XAdapter build_inputs
完成日期：2026-05-31
修改文件：`slam/models/pi3x_adapter.py`, `tests/slam/test_pi3x_adapter.py`, `docs/slam_unified_task_queue.md`
完成内容：使用 `build_view_dict(include_priors=True)` 构造 Pi3X view list；保留 `camera_intrinsics/intrinsics/ray_directions/camera_pose/camera_poses/depthmap/depth_z`；depth prior channel 被压缩到模型更常用的 HxW 格式；支持 image-only 和 prior 模式。
测试命令：`python -m pytest tests/slam/test_pi3x_adapter.py -q`

### T302 — [DONE] 实现 Pi3XAdapter 真实推理与输出标准化
完成日期：2026-06-01
修改文件：`slam/models/pi3x_adapter.py`, `configs/slam/pi3x_folder.yaml`, `tests/slam/test_pi3x_adapter.py`, `tests/slam/test_real_image_local_weights.py`, `docs/slam_unified_task_queue.md`
完成内容：支持 auto_load、infer/callable/forward/predict、`torch.no_grad()`、可选 autocast、nested raw output unwrap；标准化常见 Pi3X/MapAnything-like 输出字段，3x4 pose 自动补成 4x4，depth/confidence squeeze；真实 Pi3X wrapper 要求 `data_norm_type=identity`，已同步到 adapter 默认值与 folder 配置。
测试命令：`python -m pytest tests/slam/test_pi3x_adapter.py tests/slam/test_pi3x_folder_config.py -q`
真实环境测试命令：`source /opt/conda/etc/profile.d/conda.sh && conda activate mapanything && SLAM_RUN_REAL_IMAGE_TESTS=1 python -m pytest tests/slam/test_real_image_local_weights.py -q -s`
真实环境测试结果：`pi3x-30-one-chunk-rrd` 和 `pi3x-30-overlap` 通过；输出 `outputs/slam/real_image_local_weights/pi3x_30f_chunk30_no_opt/pi3x_30f_chunk30_no_opt.rrd` 与 `outputs/slam/real_image_local_weights/pi3x_30f_chunk8_overlap2_no_opt/pi3x_30f_chunk8_overlap2_no_opt.rrd`。
遗留问题：尚未做 Pi3X 更长序列/backend 组合真实评估。

### T303 — [DONE] Pi3X 单 chunk prior 测试
完成日期：2026-05-31
修改文件：`tests/slam/test_pi3x_adapter.py`, `tests/slam/test_pi3x_folder_config.py`, `docs/slam_unified_task_queue.md`
测试命令：`python -m pytest tests/slam/test_pi3x_adapter.py tests/slam/test_pi3x_folder_config.py -q`

### T304 — [DONE] Pi3X 多 chunk depth prior propagation 测试
完成日期：2026-05-31
修改文件：`tests/slam/test_pi3x_multichunk_priors.py`, `docs/slam_unified_task_queue.md`
测试命令：`python -m pytest tests/slam/test_pi3x_multichunk_priors.py -q`
真实环境测试结果：Pi3X 30 帧、`chunk_size=8`、`overlap=2` 端到端通过；输出 `outputs/slam/real_image_local_weights/pi3x_30f_chunk8_overlap2_no_opt/`。
遗留问题：更长序列/更多 overlap 组合仍待压力测试。

---

## P4. MapAnything 路线

### T400 — [DONE] 完善 MapAnythingAdapter
完成日期：2026-06-01
修改文件：`slam/models/mapanything_adapter.py`, `tests/slam/test_mapanything_adapter.py`, `tests/slam/test_real_wrapper_output_standardization.py`, `tests/slam/test_real_image_local_weights.py`, `docs/slam_unified_task_queue.md`
完成内容：MapAnything inference 前移除真实 `validate_input_views_for_inference` 不接受的 `camera_pose/camera_intrinsics/depthmap` 别名，保留 `camera_poses/intrinsics/depth_z`。
测试命令：`python -m pytest tests/slam/test_mapanything_adapter.py tests/slam/test_real_wrapper_output_standardization.py -q`
真实环境测试结果：`mapanything-v1-30-one-chunk-rrd` 和 `mapanything-v1-30-overlap` 通过；输出 `outputs/slam/real_image_local_weights/mapanything_v1_30f_chunk30_no_opt/mapanything_v1_30f_chunk30_no_opt.rrd` 与 `outputs/slam/real_image_local_weights/mapanything_v1_30f_chunk8_overlap2_no_opt/mapanything_v1_30f_chunk8_overlap2_no_opt.rrd`。

### T401 — [DONE] 新增 MapAnything folder 配置
修改文件：`configs/slam/mapanything_folder.yaml`, `tests/slam/test_mapanything_folder_config.py`
测试命令：`python -m pytest tests/slam/test_mapanything_folder_config.py -q`

### T402 — [DONE] MapAnything 单 chunk 多先验组合测试
修改文件：`tests/slam/test_mapanything_adapter.py`
测试命令：`python -m pytest tests/slam/test_mapanything_adapter.py -q`

### T403 — [DONE] MapAnything 多 chunk overlap depth prior 测试
修改文件：`tests/slam/test_mapanything_multichunk_priors.py`
测试命令：`python -m pytest tests/slam/test_mapanything_multichunk_priors.py -q`
真实环境测试结果：MapAnything v1 30 帧、`chunk_size=8`、`overlap=2` 端到端通过；输出 `outputs/slam/real_image_local_weights/mapanything_v1_30f_chunk8_overlap2_no_opt/`。

### T404 — [DONE] MapAnything + world_translation Sim3 对齐测试
修改文件：`tests/slam/test_mapanything_multichunk_priors.py`
测试命令：`python -m pytest tests/slam/test_mapanything_multichunk_priors.py -q`

---

## P5. 后端优化路线

### T500 — [PARTIAL] 实现 GTSAM-first SE3PoseGraphBackend 真实优化
最近更新：2026-06-01
修改文件：`slam/backend/se3_pose_graph.py`, `tests/slam/test_se3_pose_graph_backend.py`, `tests/slam/test_real_image_local_weights.py`, `docs/slam_unified_task_queue.md`
完成内容：GTSAM-first Pose3 factor graph；无 GTSAM 环境保留 scipy/numpy translation fallback；真实 VGGT 8 帧单 chunk 与 30 帧多 chunk backend 端到端 smoke 已通过并保存 RRD。
测试命令：`python -m pytest tests/slam/test_se3_pose_graph_backend.py -q`
真实环境测试命令：`source /opt/conda/etc/profile.d/conda.sh && conda activate mapanything && SLAM_RUN_REAL_IMAGE_TESTS=1 python -m pytest tests/slam/test_real_image_local_weights.py -q -s`
真实环境测试结果：`vggt-se3` 与 `vggt-se3-30-overlap` 通过；输出 `outputs/slam/real_image_local_weights/vggt_se3_pose_graph_8f/vggt_se3_pose_graph_8f.rrd` 与 `outputs/slam/real_image_local_weights/vggt_se3_pose_graph_30f_chunk8_overlap2/vggt_se3_pose_graph_30f_chunk8_overlap2.rrd`。
遗留问题：GTSAM 环境下真实分支、较长真实序列端到端效果仍需补测。

### T501 — [PARTIAL] 实现 Sim3PoseGraphBackend 真实优化
最近更新：2026-06-01
修改文件：`slam/backend/sim3_pose_graph.py`, `tests/slam/test_sim3_pose_graph_backend.py`, `tests/slam/test_real_image_local_weights.py`, `docs/slam_unified_task_queue.md`
完成内容：chunk-level Sim(3) anchor graph；支持 world_translation / overlap_pose / world_rotation；输出 de-duplicated global trajectory；真实 Pi3 8 帧单 chunk backend smoke、Pi3 30 帧多 chunk backend overlap 已通过并保存 RRD。
测试命令：`python -m pytest tests/slam/test_sim3_pose_graph_backend.py -q`
真实环境测试结果：`pi3-sim3` 与 `pi3-sim3-30-overlap` 通过；输出 `outputs/slam/real_image_local_weights/pi3_sim3_pose_graph_8f/pi3_sim3_pose_graph_8f.rrd` 与 `outputs/slam/real_image_local_weights/pi3_sim3_pose_graph_30f_chunk8_overlap2/pi3_sim3_pose_graph_30f_chunk8_overlap2.rrd`。
遗留问题：当前是 per-chunk anchor refinement，不是完整 joint Sim3 nonlinear graph；dense pointmap matching factors 和更长真实序列尚未实现/补测。

### T502 — [PARTIAL] SensorFusionGraphBackend
最近更新：2026-06-01
修改文件：`slam/backend/sensor_fusion_graph.py`, `scripts/slam/run_slam_folder.py`, `tests/slam/test_sensor_fusion_graph_backend.py`, `tests/slam/test_real_image_local_weights.py`, `docs/slam_unified_task_queue.md`
完成内容：新增 GTSAM-first sensor fusion shell；支持预处理后的 `imu_relative_pose / sensor_relative_pose / wheel_relative_pose / gnss_relative_pose`；raw IMU preintegration 显式 unsupported；runner 支持 `--backend_type sensor_fusion_graph`；真实 VGGT 8 帧单 chunk 与 30 帧多 chunk 无 raw sensor smoke 已通过并保存 RRD。
测试命令：`python -m pytest tests/slam/test_sensor_fusion_graph_backend.py -q`
真实环境测试结果：`vggt-sensor-fusion` 与 `vggt-sensor-fusion-30-overlap` 通过；输出 `outputs/slam/real_image_local_weights/vggt_sensor_fusion_graph_8f/vggt_sensor_fusion_graph_8f.rrd` 与 `outputs/slam/real_image_local_weights/vggt_sensor_fusion_graph_30f_chunk8_overlap2/vggt_sensor_fusion_graph_30f_chunk8_overlap2.rrd`。
遗留问题：raw IMU preintegration 需要扩展 SlamFrame/BackendPrior，加入 velocity、bias、IMU measurement buffer；真实 sensor factor 数据待接入。

### T503 — [PARTIAL] SL4SubmapGraphBackend
最近更新：2026-06-01
修改文件：`slam/backend/sl4_submap_graph.py`, `scripts/slam/run_slam_folder.py`, `tests/slam/test_sl4_submap_graph_backend.py`, `tests/slam/test_real_image_local_weights.py`, `docs/slam_unified_task_queue.md`
完成内容：chunk/submap-level SL4 backend；GTSAM 分支使用 `gtsam.SL4 / PriorFactorSL4 / BetweenFactorSL4 / LM`；无 gtsam.SL4 环境使用 deterministic fallback；真实 VGGT 8 帧单 chunk 与 30 帧多 chunk fallback smoke 已通过并保存 RRD。
测试命令：`python -m pytest tests/slam/test_sl4_submap_graph_backend.py -q`
真实环境测试结果：`vggt-sl4` 与 `vggt-sl4-30-overlap` 通过；输出 `outputs/slam/real_image_local_weights/vggt_sl4_submap_graph_8f/vggt_sl4_submap_graph_8f.rrd` 与 `outputs/slam/real_image_local_weights/vggt_sl4_submap_graph_30f_chunk8_overlap2/vggt_sl4_submap_graph_30f_chunk8_overlap2.rrd`。
遗留问题：真实 gtsam-develop / SL4 环境、attention loop verification / auto-calibration homography 待补测。

---

## P6. 地图与输出路线

### T600 — [PARTIAL] 完善真实输出保存
最近更新：2026-06-04
修改文件：

```text
slam/io/export_outputs.py
scripts/slam/validate_slam_exports.py
scripts/slam/run_slam_folder.py
tests/slam/test_export_outputs.py
tests/slam/test_validate_slam_exports_script.py
tests/slam/test_run_slam_folder_script.py
tests/slam/test_real_image_local_weights.py
docs/slam_unified_task_queue.md
```

完成内容：

```text
1. 新增 reusable exporter：save_slam_outputs；
2. 支持 summary.json 与 export_manifest.json；
3. 支持 trajectory.npz / 显式 --output_npz；
4. 支持 trajectory.csv；
5. 支持 TUM trajectory：trajectory_tum.txt，可用于 evo 风格评估；
6. 支持 cameras.json，包含预测位姿、GT 位姿和 intrinsics；
7. 支持 points.npy；
8. 支持 ASCII PLY point cloud，可带图像颜色；
9. 保留可选 RRD 输出与 sidecar json；
10. run_slam_folder.py 改为调用统一 exporter，避免重复维护导出逻辑；
11. 支持 COLMAP text sparse model：`colmap_text/cameras.txt`, `colmap_text/images.txt`, `colmap_text/points3D.txt`；
12. 导出 summary/manifest 记录 `world_prediction_chunks`，用于验证真实 multi chunk 是否实际发生；
13. 真实 VGGT/Pi3/Pi3X/MapAnything 30 帧单 chunk 与 30 帧 `chunk_size=8` / `overlap=2` multi chunk 已验证 summary/manifest/npz/csv/tum/json/npy/ply/rrd/COLMAP text 全部写出；
14. 支持逐帧预测数组导出：`prediction_arrays.json`, `depth/*.npy`, `confidence/*.npy`, `valid_mask/*.npy`，用于调试稠密深度/置信度/有效 mask；
15. 新增 `scripts/slam/validate_slam_exports.py`，可人工验证导出目录中的 summary/manifest/npz/TUM/cameras/COLMAP/PLY/prediction arrays，并报告 Open3D/Rerun/evo/pycolmap 可选工具可用性；
16. 支持每个 chunk 的 prediction debug 包：`chunk_predictions/index.json` 与 `chunk_XXXXXX/{world,raw,local}.npz`，其中 world 保存对齐后的 `T_world_cam/points_world/depth/intrinsics/confidence/valid_mask/T_world_model`，raw/local 保存模型坐标侧 `T_model_cam/points_model/depth/intrinsics/confidence/valid_mask`、模型名和 diagnostics；当 raw 本身就是 local 坐标时，index 中 `local` 以 alias 形式指向 raw 包，避免重复写大数组。
17. 2026-06-04 对既有真实输出做离线 core export validation：覆盖 VGGT 单 chunk、VGGT/Pi3/Pi3X/MapAnything 30 帧 multi chunk，以及 VGGT+SE3、Pi3+Sim3、VGGT+SensorFusion、VGGT+SL4 backend multi chunk，共 9 个目录；summary/manifest/显式 trajectory npz/csv/TUM/cameras.json/COLMAP text/points.npy/ASCII PLY 均通过结构与可读性检查。验证报告保存到 `outputs/slam/export_validation_reports/core_2026_06_04/summary.json`；该目录在当前 repo 中属于 ignored outputs。
```

参考/复用来源：

```text
scripts/predict_scene_to_rrd.py：Rerun/RRD 可视化布局和相机轴调试思路；
Open3D / COLMAP / evo 常用格式：PLY point cloud、TUM trajectory、camera json/csv；
numpy/json/yaml 风格：轻量可测试导出。
```

是否直接复用外部代码：否。RRD 布局参考本项目已有脚本；无外部代码复制。

测试命令：

```bash
python -m pytest tests/slam/test_export_outputs.py tests/slam/test_run_slam_folder_script.py -q
python3 -m pytest tests/slam/test_validate_slam_exports_script.py -q
python3 -m pytest tests/slam/test_export_outputs.py tests/slam/test_validate_slam_exports_script.py -q
python3 -m pytest tests/slam/test_real_image_local_weights.py -q
python3 -m pytest tests/slam -q
source /opt/conda/etc/profile.d/conda.sh && conda activate mapanything && SLAM_RUN_REAL_IMAGE_TESTS=1 python -m pytest tests/slam/test_real_image_local_weights.py -q -s
python3 scripts/slam/validate_slam_exports.py --output_dir outputs/slam/real_image_local_weights/vggt_30f_chunk8_overlap2_no_opt --no_prediction_arrays
python3 - <<'PY'
from pathlib import Path
import json
from scripts.slam.validate_slam_exports import validate_export_dir

root = Path('outputs/slam/real_image_local_weights')
selected = [
    'vggt_30f_chunk30_no_opt',
    'vggt_30f_chunk8_overlap2_no_opt',
    'pi3_30f_chunk8_overlap2_no_opt',
    'pi3x_30f_chunk8_overlap2_no_opt',
    'mapanything_v1_30f_chunk8_overlap2_no_opt',
    'vggt_se3_pose_graph_30f_chunk8_overlap2',
    'pi3_sim3_pose_graph_30f_chunk8_overlap2',
    'vggt_sensor_fusion_graph_30f_chunk8_overlap2',
    'vggt_sl4_submap_graph_30f_chunk8_overlap2',
]
report_root = Path('outputs/slam/export_validation_reports/core_2026_06_04')
report_root.mkdir(parents=True, exist_ok=True)
summary = []
for name in selected:
    report = validate_export_dir(root / name, require_prediction_arrays=False)
    report_path = report_root / f'{name}.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    summary.append({'name': name, 'ok': report['ok'], 'errors': report['errors'], 'warnings': report['warnings'], 'counts': report.get('counts', {}), 'optional_tools': report.get('optional_tools', {}), 'report': str(report_path)})
summary_path = report_root / 'summary.json'
summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
if not all(item['ok'] for item in summary):
    raise SystemExit(1)
PY
```

测试状态：真实本地权重/真实图像测试已通过；RRD 与 COLMAP text 文件已保存到 `outputs/slam/real_image_local_weights/`。2026-06-04 离线 core export validation 覆盖 9 个既有真实输出目录，结果均为 `ok=true`、`errors=0`；每个目录因既有输出缺少 `chunk_predictions/index.json` 产生 1 条 warning，逐帧 prediction arrays 在本次命令中按旧输出兼容模式跳过。
单元测试结果：`113 passed, 19 skipped in 1.58s`；chunk prediction debug 小集 `4 passed in 0.33s`
真实环境测试结果：基础/模型 multichunk 全量 `12 passed in 659.26s`；backend multichunk 子集 `4 passed, 12 deselected in 294.67s`

无法在当前环境完成的测试：

```text
1. Rerun viewer 人工打开 .rrd 检查；
2. evo 对 trajectory_tum.txt 的真实评估；当前验证脚本仅检查格式与可选模块可用性；
3. COLMAP/OpenMVS 对 `colmap_text/*.txt` 的真实读取检查；当前验证脚本仅做文本结构检查与 pycolmap 可用性报告；
4. Open3D / MeshLab / CloudCompare 打开 points.ply 的真实检查；当前验证脚本仅做 PLY header/点数基础检查与 Open3D 可用性报告；
5. 真实 VGGT/Pi3/Pi3X/MapAnything 更长序列 / 更大 chunk / backend 组合输出规模压力测试；
6. 当前轻量环境未安装 `open3d` / `rerun` / `evo` / `pycolmap` Python 模块，`--strict_optional_tools` 读取验证无法通过；需在工具链环境补测；
7. 2026-06-04 验证的既有真实输出目录生成于 prediction arrays / chunk prediction debug 包真实重导出之前，因此本次仅验证 core export；完整 debug 包真实输出仍需重新跑真实模型后补测。
```

遗留问题：

```text
1. COLMAP text 当前为 sparse/debug 兼容格式，`points3D.txt` 默认采样上限避免 dense map 文件过大；
2. PLY 当前为 ASCII，超大点云后续可增加 binary PLY/Open3D writer；
3. 当前逐帧深度/置信度导出为 `.npy`，后续真实工具链需要时可增加 EXR/PNG 可视化副本；
4. chunk prediction debug 包已纳入轻量 validator，但尚未在真实长序列输出规模下做磁盘占用和加载性能评估；
5. 既有真实输出通过 core export validation，但缺少 chunk prediction debug 包；后续真实重导出后应重新运行默认 `require_prediction_arrays=True` 的 validator。
```

下一步：人工打开 RRD 检查尺度/axes 观感；在安装 `open3d` / `rerun` / `evo` / `pycolmap` 或外部 COLMAP/OpenMVS 的环境读取真实导出文件；重新跑真实模型生成 prediction arrays / chunk prediction debug 包并做默认 validator；继续真实多 chunk/长序列测试。

---

### T601 — [DEFERRED] DEMMap 真实网格融合
参考/复用来源：VGGT-SLAM++ DEM tile / covisibility map、Open3D、rasterio、本项目 UAV/DOM 输出需求。

### T602 — [DEFERRED] GaussianMap / RenderingBackend
参考/复用来源：SplaTAM、ARTDECO、M3 的 Gaussian map / differentiable rendering / camera pose + Gaussian joint optimization。

---

## P7. Loop closure 路线

### T700 — [DONE] Loop retrieval 接口
最近更新：2026-06-01
修改文件：`slam/loop/retrieval.py`, `slam/loop/dino_salad_provider.py`, `slam/loop/loop_manager.py`, `tests/slam/test_generic_loop_closure.py`, `tests/slam/test_dino_salad_embedding_retrieval.py`, `docs/slam_unified_task_queue.md`
完成内容：已有 trajectory-distance baseline retrieval、provider-backed embedding retrieval、DINO/SALAD provider shell；无 torch 环境下支持 injected lightweight model 的 numpy fallback，便于 CI。
测试命令：`python3 -m pytest tests/slam/test_generic_loop_closure.py tests/slam/test_dino_salad_embedding_retrieval.py -q`
测试结果：通过。
参考/复用来源：MASt3R-SLAM retrieval / loop closure、VGGT-SLAM 2.0 attention features、VGGT-SLAM++ DINOv2 embedding + DEM/covisibility、ORB-SLAM3 place recognition。
遗留问题：真实 DINO/SALAD checkpoint 检索、tile/DEM/covisibility retrieval 尚未补测。

### T701 — [PARTIAL] Geometric verifier
最近更新：2026-06-03
修改文件：`slam/loop/geometric_verifier.py`, `scripts/slam/run_slam_folder.py`, `configs/slam/generic_dummy.yaml`, `tests/slam/test_generic_loop_closure.py`, `tests/slam/test_pointcloud_registration_verifier.py`, `tests/slam/test_pnp_ransac_verifier.py`, `tests/slam/test_run_slam_folder_script.py`, `docs/slam_unified_task_queue.md`
完成内容：已有 pose-distance verifier baseline，可生成 SE3 verified loop。新增 `PointCloudRegistrationVerifier`，支持从 `LoopCandidate.metadata` 读取 `match_points/query_points`，或由 loop manager 传入 `frames + adapter` 后重跑 `[match_frame, query_frame]` pair chunk 获取标准化 `points_model`；支持 SE3/Sim3 Umeyama、nearest-neighbor inlier/RMSE 判定，以及可选 Open3D ICP refine。新增 `PnPRansacVerifier`，支持 `object_points + image_points + intrinsics` 的 2D-3D loop verification，优先使用 OpenCV `solvePnPRansac`，无 OpenCV 时使用 numpy DLT/RANSAC fallback，输出与 loop factor 一致的 `T_match_query`。`run_slam_folder.py` 可通过 `--loop_verifier pointcloud_registration` 或 `--loop_verifier pnp_ransac` 和对应阈值参数直接启用 verifier。
测试命令：`python3 -m pytest tests/slam/test_pointcloud_registration_verifier.py tests/slam/test_pnp_ransac_verifier.py tests/slam/test_run_slam_folder_script.py tests/slam/test_generic_loop_closure.py -q`; `python3 -m pytest tests/slam -q`
测试结果：相关测试通过；全量 `113 passed, 19 skipped in 1.58s`
参考/复用来源：MASt3R pointmap matching、VGGT tracks/attention/point maps、OpenCV PnP/RANSAC、Open3D registration。
遗留问题：真实大模型 pointmap matching、OpenCV/Open3D 真环境与长序列 loop closure 尚未补测。

### T702 — [PARTIAL] Loop factor 接入后端
最近更新：2026-06-03
修改文件：`slam/loop/factor_adapter.py`, `slam/core/generic_slam.py`, `slam/backend/se3_pose_graph.py`, `slam/backend/no_opt_backend.py`, `scripts/slam/run_slam_folder.py`, `configs/slam/generic_dummy.yaml`, `tests/slam/test_generic_loop_closure.py`, `tests/slam/test_run_slam_folder_script.py`, `docs/slam_unified_task_queue.md`
完成内容：LoopFactorAdapter 可生成 SE3 loop、Sim3 overlap pose、SL4 between factor；GenericFeedForwardSLAM 可选 loop_manager 后把 verified loop factors 写回 backend；SE3 fallback 兼容 loop overlap pose 的 `T_world_cam`。`run_slam_folder.py` 已支持通过 config/CLI 打开 loop closure，并可在 NoOptBackend 下记录 loop factors 做调试。
测试命令：`python3 -m pytest tests/slam/test_generic_loop_closure.py tests/slam/test_run_slam_folder_script.py -q`; `python3 -m pytest tests/slam -q`
测试结果：全量 `113 passed, 19 skipped in 1.58s`
参考/复用来源：VGGT-SLAM / VGGT-SLAM 2.0 / MASt3R-SLAM / ORB-SLAM3 / GTSAM。
遗留问题：真实 long-sequence loop closure、dense matching loop factor、真 GTSAM loop optimization 待补测。

---

# 5. 当前最推荐立即执行的任务

```text
1. 人工打开 `outputs/slam/real_image_local_weights/` 下的 RRD，检查 pred/gt axes 尺度和可视化观感；
2. 真实模型更长序列 / 更大 chunk / backend 组合跑通并检查 T600 导出；
3. 在 gtsam-develop 环境验证 Pose3 与 SL4 真 GTSAM 分支；
4. 扩展 T701/T702: dense/geometric loop verification 与真实长序列 loop factor；
5. 用 COLMAP/OpenMVS/Open3D/evo/Rerun viewer 读取真实导出文件并记录结果。
```

不要优先执行：DEMMap 真实融合、GeoFF3D。

---

# 6. 每次任务完成后的汇报模板

```text
任务编号：
当前状态：
完成日期：
完成内容：
修改文件：
新增/更新测试：
参考/复用来源：
是否直接复用外部代码：是/否
若直接复用，来源注释位置：
测试命令：
测试结果：
无法在当前环境完成的测试：
后续验证命令：
遗留问题：
已更新任务队列：是/否
下一步建议：
```

如果“已更新任务队列”为“否”，则该任务视为未完成。
