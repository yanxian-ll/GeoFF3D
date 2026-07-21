# Generic Feed-forward SLAM 技术节点设计说明

本文档是当前 Generic Feed-forward SLAM 框架的**唯一推荐技术设计入口**。它把旧 `slam_design.md`、loop closure 调研文档、pair inference 设计文档中的有效内容合并到同一个按代码抽象组织的说明中。

这里的“技术节点”不是论文名，而是代码中的可替换模块：

```text
Data Node: Frame / Chunk / Prediction
Frontend Node: Chunking / Keyframe Selection
Model Adapter Node: ModelSpec / build_inputs / infer / standardize
Prior Node: ModelInputPrior / BackendPrior
Coordinate Normalization Node: model/local/projective → world
Overlap Node: adjacent chunk pose/depth propagation
Loop Closure Node: Retrieval → Verification → Factor
Backend Node: SE3 / Sim3 / SL4 / Dense GN / SensorFusion
Mapping Node: PointCloud / DEM / Gaussian
Exporter / Diagnostics
```

目标：

```text
1. 让 GenericFeedForwardSLAM 主流程不绑定任何具体模型；
2. 让 VGGT / Pi3 / Pi3X / MapAnything / MASt3R / ARTDECO / M³ 等方法映射到统一抽象；
3. 明确哪些功能是当前已实现，哪些是 provider / verifier / backend plugin；
4. 保留开源方法中有价值的设计，但不把模型专用逻辑写死到主流程。
```

---

# 0. 总体流程

当前 `GenericFeedForwardSLAM.run()` 的抽象流程：

```text
frames
→ ChunkManager.iter_chunks
→ OverlapManager.populate_depth_priors
→ PriorBuilder.build
→ adapter.infer(chunk, model_prior)
→ CoordinateNormalizer.to_world
→ backend.add_chunk
→ mapping.integrate
→ OverlapManager.update
→ backend.optimize
→ optional loop_manager.run
→ append loop factors
→ backend.optimize again
→ trajectory / map / diagnostics / outputs
```

对应代码：

```text
slam/core/generic_slam.py
slam/core/data_types.py
slam/models/model_spec.py
slam/frontend/chunk_manager.py
slam/frontend/overlap_manager.py
slam/priors/prior_builder.py
slam/geometry/coordinate_normalizer.py
slam/loop/*
slam/backend/*
slam/mapping/*
slam/io/export_outputs.py
```

核心边界：

```text
1. 模型只通过 adapter 接入；
2. normalizer 负责把模型坐标系对齐到 world/map frame；
3. backend 负责跨 chunk、loop、sensor factor 优化；
4. mapping 负责地图融合，不反向污染前端主流程；
5. loop closure 必须拆成 retrieval → verification → factor；
6. 具体开源项目只能替换某个技术节点，不能污染 GenericFeedForwardSLAM。
```

---

# 1. Data Node：Frame / Chunk / Prediction

## 1.1 当前代码设计

当前核心数据结构：

```text
SlamFrame
Chunk
StandardChunkPrediction
WorldChunkPrediction
```

### SlamFrame

```text
frame_id
timestamp
image
intrinsics
ray_directions
world_translation
world_rotation
depth_prior
depth_mask
metadata
```

含义：输入帧 + 可选传感器/几何先验。是否使用这些先验由 `ModelSpec` 和 adapter 决定。

### Chunk

```text
chunk_id
frames
overlap_frame_ids
```

含义：feed-forward 模型的一次输入窗口。VGGT / Pi3 / Pi3X / MapAnything 都可以包装成“输入一个 chunk，输出一个 chunk prediction”。

### StandardChunkPrediction

```text
frame_ids
T_model_cam
intrinsics
depth
points_model
confidence
valid_mask
tracks
matching_features
attention_features
coord_type
scale_type
model_name
diagnostics
```

关键约定：

```text
T_model_cam / points_model 位于模型自己的坐标系；
coord_type 标记坐标系含义：world / metric_local / projective / local；
scale_type 标记尺度含义：metric / up_to_scale / unknown；
tracks / matching_features / attention_features 是给 tracking / loop / verifier 使用的可选字段。
```

### WorldChunkPrediction

```text
frame_ids
T_world_cam
points_world
depth
intrinsics
confidence
valid_mask
T_world_model
alignment_type
raw_prediction
diagnostics
```

含义：`CoordinateNormalizer` 之后的结果，用于 backend 和 mapping。

## 1.2 不同方法映射

### DUSt3R / MASt3R：pointmap 作为统一几何中间表示

DUSt3R / MASt3R 的核心启发是：不要只把模型输出当 pose，而要把 dense pointmap 作为 SLAM 的中间表示。

```text
DUSt3R:
  image pair / multi-image → pointmap
  多图像再通过 global alignment 放到共同坐标系

MASt3R:
  pointmap + dense local feature / matching head
  支持快速 reciprocal matching
```

映射到本项目：

```text
StandardChunkPrediction.points_model 必须是核心字段；
StandardChunkPrediction.confidence 用于过滤点云/匹配；
StandardChunkPrediction.matching_features 用于 MASt3R-style retrieval / dense verifier；
Loop verifier 不应只依赖 pose，也可以依赖 pointmap matching。
```

### VGGT / VGGT-Omega：一次前馈输出完整几何包

```text
输入：多帧图像；可选 intrinsics/rays 视 wrapper 而定。
输出：camera / depth / point map / confidence / 3D tracks。
坐标系：通常不是稳定 world，适合 submap/projective 或 metric-local 对齐。
```

映射：

```text
T_model_cam
intrinsics
depth
points_model
confidence
tracks
coord_type="local" 或 "projective"
scale_type="unknown" 或 "up_to_scale"
```

注意：VGGT tracks 可以用于 overlap frame 之外的 long-range association，因此后续可用于：

```text
TrackBasedLoopRetrieval
TrackPnPVerifier
cross-chunk association
```

### Pi3

```text
输入：多帧图像。
输出：pose / depth / point map / confidence。
坐标系：local / metric_local，需要 overlap 或 world priors 对齐。
推荐后端：Sim3 或 SE3，取决于尺度稳定性。
```

### Pi3X / MapAnything

```text
输入：图像 + 可选 intrinsics / rays / pose / translation / rotation / depth prior / partial reconstruction。
输出：metric_local 几何。
坐标系：可借助先验更接近 world，但模型输出本身不应默认等于 world。
```

映射：

```text
PriorBuilder 负责把 SlamFrame priors 转成模型输入；
CoordinateNormalizer 负责 local-to-world 对齐；
BackendPrior 负责把这些先验也加入后端优化；
不能把“模型吃了先验”误认为“后端已经有约束”。
```

### MASt3R-SLAM / ARTDECO

它们更 keyframe-centric，而不是简单 chunk-centric：

```text
ImageFrame / KeyFrame:
  img
  feat
  pos
  X_canon / pointmap
  C / confidence
  T_WC / Sim3 pose
```

映射：

```text
StandardChunkPrediction.points_model = X_canon
StandardChunkPrediction.confidence = C
StandardChunkPrediction.matching_features = (feat, pos)
```

dense matching verifier 和 dense GN backend 不应写进 adapter，而应作为：

```text
Mast3RRetrievalProvider
Mast3RDensePairVerifier
DenseGNBackend
```

### VGGT-SLAM++

其核心数据不只是 frame，而是：

```text
VGGT submap
DEM tile
DINOv2 tile embedding
covisibility graph node
```

映射：

```text
DEMMapTile
TileLoopCandidate
TileLoopRetrieval
TileOverlapVerifier
MappingCorrectionRequest
```

### M³ / Pi3X multi-view

M³ / Pi3X multi-view 路线的候选不是单 pair，而是一组 historical keyframes：

```text
query_frame_ids
history_frame_ids
multi-view inference result
matching head correspondences
```

映射：

```text
MultiFrameLoopCandidate
MultiViewModelVerifier
Pi3X retrieved-keyframe wrapper
```

### CUT3R / Spann3R：stateful / persistent memory model

CUT3R / Spann3R 说明有些前馈模型不是 stateless chunk inference，而是 streaming/stateful inference：

```text
每来一帧更新 persistent memory / spatial memory；
模型直接输出 common coordinate / global coordinate 下的 pointmap；
不一定需要传统 global alignment，但需要管理 memory state。
```

映射：

```text
ModelSpec.is_stateful = True
adapter.reset_state()
adapter 内部维护 memory_state
GenericFeedForwardSLAM 不应假设所有模型都是 stateless chunk model
```

---

# 2. Frontend Node：Chunking / Keyframe Selection

## 2.1 当前代码设计

当前 `ChunkManager` 是 fixed-window：

```text
chunk_size
overlap
min_chunk_size
step = chunk_size - overlap
```

适合先跑通所有模型的通用入口。

## 2.2 不同方法设计

### VGGT-SLAM 2.0

```text
以 submap 为基本单位；
每个 submap 包含连续 frames；
submap 之间通过 graph edge 连接；
loop closure 是 submap-level edge；
后端节点是 SL4 homography。
```

映射：

```text
Chunk ≈ VGGT submap
chunk_id ≈ submap_id
frame_to_chunk 用于 LoopCandidate → SL4 factor
```

### MASt3R-SLAM

```text
以 keyframe 为基本单位；
frontend 决定是否生成 keyframe；
连续 keyframe 默认加 dense matching edge；
retrieval keyframe 作为 loop edge 候选。
```

映射：

```text
Chunk 可以退化为单 keyframe 或小窗口；
后续可新增 KeyframeManager；
FactorGraph.add_factors 同时接连续边和 loop 边。
```

### ARTDECO

```text
frontend 生成 keyframe；
backend 接收 keyframe_style；
style=0 relocalization；
style=1 global optimization；
style=2 map keyframe；
mapper 消费 densePoint 和 loop_keyframe_index。
```

映射：

```text
前端 keyframe selection 和 mapper 分离；
loop_keyframe_index 应作为 mapping metadata，而不是让 mapper 反向决定 loop。
```

### VGGT-SLAM++

```text
先生成 VGGT submap；
再把 submap 转成 DEM tile；
frontend 不只是 temporal chunk，也包含 spatial/covisibility window。
```

映射：

```text
ChunkManager 保留；
后续新增 SpatialWindow / DEMTileManager。
```

### M³ / Pi3X

```text
当前窗口 + retrieved historical keyframe set 一起作为 multi-view input；
输入大小不是固定连续窗口；
keyframe selection 和 retrieval 共同决定模型输入。
```

映射：

```text
ChunkManager 负责连续 query frames；
LoopRetrieval 负责 history frames；
MultiViewModelVerifier 构造 query + history mixed chunk。
```

---

# 3. Model Adapter Node：ModelSpec / build_inputs / infer / standardize

## 3.1 当前代码设计

`ModelSpec` 描述模型能力：

```text
accepts_intrinsics
accepts_rays
accepts_pose_prior
accepts_translation_prior
accepts_rotation_prior
accepts_depth_prior
accepts_partial_points
is_stateful
predicts_camera
predicts_intrinsics
predicts_depth
predicts_points
predicts_tracks
predicts_matching_features
predicts_confidence
output_coord_type
output_scale_type
pose_convention
```

作用：

```text
PriorBuilder 根据 spec 选择传哪些先验；
CoordinateNormalizer 根据 coord_type/scale_type 决定如何对齐；
Loop verifier 根据 tracks / matching_features / points_model 决定能否做高级验证。
```

## 3.2 不同方法设计

### VGGT Adapter

```text
build_inputs:
  图像 resize / normalize；
  可选 intrinsics / rays；

infer:
  model(images)；
  pair verifier 时可 model(lc_frames, compute_similarity=True)；

standardize:
  pose encoding → extrinsic / intrinsic；
  depth / points / confidence / tracks；
  diagnostics 可包含 image_match_ratio。
```

设计边界：

```text
VGGT pair verifier 不应该访问 raw VGGT output；
adapter 应把 image_match_ratio 放进 diagnostics；
ModelPairReprojectionVerifier 可读取 diagnostics，但不硬编码 VGGT。
```

### Pi3 Adapter

```text
build_inputs:
  图像；

infer:
  multi-view point / pose prediction；

standardize:
  T_model_cam / points_model / confidence；
  coord_type 通常 local 或 metric_local；
  需要 overlap / world_translation 对齐。
```

### Pi3X / MapAnything Adapter

```text
build_inputs:
  图像；
  intrinsics / ray_directions；
  pose prior；
  translation prior；
  rotation prior；
  depth prior；
  partial points / partial reconstruction；

infer:
  用 priors 改善 metric consistency；

standardize:
  输出 metric_local；
  不默认等于 world。
```

关键原则：

```text
带先验方法也要走 CoordinateNormalizer；
同一份 prior 既要能作为 model input，也要能作为 backend factor；
adapter 内只做模型调用和输出标准化，不做全局图优化。
```

### MASt3R / ARTDECO Adapter

```text
adapter:
  输出 pointmap / confidence / feature / pos；
retrieval provider:
  使用 MASt3R retrieval feature；
verifier:
  使用 MASt3R dense symmetric/asymmetric matching；
backend:
  使用 dense GN factor graph 或转成 SE3/Sim3 factors。
```

### M³ / Pi3X Multi-view Adapter

```text
adapter 支持 mixed input：query frames + retrieved history frames；
输出 refined poses / dense correspondences / confidence；
MultiViewModelVerifier 从输出中构造 constraints。
```

### Stateful Adapter：CUT3R / Spann3R 类

```text
adapter.reset_state(): 清空 persistent memory；
adapter.infer(frame_or_chunk): 更新 memory_state 并输出 common-coordinate prediction；
GenericFeedForwardSLAM 不应在 chunk 间强制 reset；
diagnostics 应记录 memory length / dropped frames / state id。
```

---

# 4. Prior Node：ModelInputPrior / BackendPrior

## 4.1 当前代码设计

`PriorBuilder` 同时构建两类 prior：

```text
ModelInputPrior:
  传给模型 forward；

BackendPrior:
  传给后端优化。
```

ModelInputPrior 可能包含：

```text
intrinsics
ray_directions
pose_prior
translation_prior
rotation_prior
depth_prior
validity_mask
partial_points
```

BackendPrior 可能收集：

```text
world_translation_factors
world_rotation_factors
intrinsics_prior_factors
depth_prior_factors
loop_factors
sensor_relative_pose_factors
```

## 4.2 不同方法设计

### 无先验方法：VGGT / Pi3

```text
ModelInputPrior 通常为空或只含 intrinsics；
BackendPrior 可来自 GNSS / IMU / cams_dir；
CoordinateNormalizer 使用 overlap 或 world_translation 对齐。
```

### 带先验方法：Pi3X / MapAnything / G-CUT3R

```text
同一 prior 有双重作用：
1. 作为 model input prior，让模型预测更稳定；
2. 作为 backend factor，约束最终 trajectory。
```

注意：

```text
不能只传给模型，不传给 backend；
不能只传给 backend，不传给模型；
先验系统不应写死某一种先验格式；
adapter.spec 必须声明可接受哪些 prior。
```

### MASt3R-Fusion：Sim3 visual constraints → metric SE3 factor graph

MASt3R-Fusion 对本项目的核心启发：

```text
feed-forward pointmap regression 产生视觉 Sim(3) 约束；
IMU / GNSS 负责把系统约束到 metric-scale SE(3)；
后端同时支持 sliding-window optimization 和 global optimization with loop closures。
```

映射：

```text
视觉模型输出不一定直接是 metric SE3；
可以先生成 Sim3 visual factor；
SensorFusionGraphBackend 最终优化 metric SE3 trajectory；
world_translation / world_rotation 是当前 GNSS/IMU pseudo factor 的最小版本；
raw IMU preintegration 后续需要 velocity / bias / IMU buffer。
```

### VGGT-SLAM 2.0

```text
prior 主要体现为 submap graph edge / SL4 factor；
不是直接输入每帧 GNSS prior；
loop pair inference 产生额外 extrinsic_lc / intrinsic_lc / depth_lc / depth_conf_lc。
```

### MASt3R-SLAM / ARTDECO

```text
连续 keyframe edge 是默认 prior；
retrieval edge 是 loop prior；
dense matching 结果是 factor data；
calib K 决定 solve_GN_calib 还是 solve_GN_rays。
```

---

# 5. Coordinate Normalization Node：model/local/projective → world

## 5.1 当前代码设计

`CoordinateNormalizer.to_world()` 根据 `pred.coord_type` 和 priors 处理：

```text
coord_type == world:
  直接作为 world 输出；

否则：
  优先用 >=3 world_translation 做 Umeyama Sim3；
  其次用 OverlapManager estimate_alignment；
  第一个 chunk 无 prior 时 identity；
  否则报错。
```

输出：

```text
T_world_model
T_world_cam
points_world
alignment_type
```

## 5.2 不同方法设计

### VGGT-SLAM / VGGT-SLAM 2.0：SL4 projective ambiguity

VGGT-like uncalibrated reconstruction 可能存在 projective ambiguity。VGGT-SLAM 的关键启发是：如果模型输出存在 15DoF projective ambiguity，仅靠 Sim3/SE3 对齐不够，应使用 SL(4) submap graph。

```text
每个 submap 一个 SL4 node；
relative_h 作为 BetweenFactorSL4；
anchor prior 固定 gauge；
Levenberg-Marquardt 优化整个 SL4 graph。
```

映射：

```text
CoordinateNormalizer 可以先生成近似 world prediction；
真正 projective consistency 交给 SL4SubmapGraphBackend；
LoopFactorAdapter 输出 sl4_between。
```

### Pi3 / Pi3X / MapAnything

```text
模型输出 metric_local；
如果有 world_translation >=3：Umeyama Sim3 对齐；
如果相邻 chunk 有 overlap：用 overlap frame pose 对齐；
如果只有单 overlap frame：用 pose anchor；
后端再优化全局一致性。
```

### MASt3R-SLAM

```text
每个 keyframe 有 canonical pointmap X_canon；
每个 keyframe 有 Sim3 pose T_WC；
GN 优化 pose 和 pointmap / dense matches；
最终 keyframes.update_T_WCs。
```

映射：

```text
CoordinateNormalizer 负责初始对齐；
DenseGNBackend 负责真实全局 refinement。
```

### CUT3R / Spann3R

```text
模型可能直接输出 common/global coordinate pointmaps；
CoordinateNormalizer 应允许 coord_type="world" 或 "common"；
但仍要记录 alignment_type，避免误判坐标系来源。
```

---

# 6. Overlap Node：相邻 chunk 的 pose/depth propagation

## 6.1 当前代码设计

`OverlapManager` 存储历史 frame 的：

```text
T_world_cam
depth
confidence
intrinsics
valid_mask
points_world
```

职责：

```text
1. update(world_pred)：保存每个 frame 的 world state；
2. populate_depth_priors(chunk)：如果新 chunk 中某帧曾出现过，则注入 depth_prior / mask / intrinsics；
3. add_backend_overlap_factors(chunk, backend_prior)：给 overlap frame 添加 overlap_pose factor 和 depth prior factor。
```

## 6.2 不同方法设计

### VGGT / Pi3 / Pi3X / MapAnything

```text
相邻 chunk 共享 overlap frames；
上一 chunk 的 depth 可作为下一 chunk 的 depth prior；
上一 chunk 的 T_world_cam 可作为下一 chunk 的 pose anchor；
如果 overlap >=3，可估计 Sim3；
如果 overlap =1，可用 pose anchor。
```

### MASt3R-SLAM

```text
连续 keyframe edge 是 overlap 的另一种形式；
不是直接传 depth prior，而是通过 dense matching factor 连接相邻 keyframes；
连续边即使 match_frac 低也更宽松。
```

### ARTDECO

```text
当前 keyframe 和 last keyframe 做 asymmetric matching；
计算 densePoint 给 mapper；
loop_keyframe_index 作为 mapper metadata。
```

### VGGT-SLAM++

```text
overlap 不局限于 temporal overlap；
DEM tile / covisibility graph 提供 spatial overlap；
local correction 由 spatial neighbors 高频触发。
```

### M³ / Pi3X

```text
overlap 可以由 retrieved historical keyframes 提供；
不是只靠连续 chunk overlap；
multi-view inference 可以把历史帧作为 context/prior。
```

---

# 7. Loop Closure Node：Retrieval → Verification → Factor

Loop closure 是当前框架最关键的扩展节点。设计原则：

```text
Retrieval 只找候选；
Verification 决定候选是否可信；
FactorAdapter 根据 backend_type 转成 SE3 / Sim3 / SL4 / dense / map correction。
```

## 7.1 Retrieval：候选生成

### 当前实现：TemporalDistanceLoopRetrieval

```text
输入：trajectory
逻辑：camera center distance + temporal exclusion
输出：LoopCandidate
优点：模型无关，CI 可跑
缺点：只能发现轨迹上已接近的闭环
```

### 当前实现：EmbeddingLoopRetrieval + DinoSaladEmbeddingProvider

```text
输入：frames
provider：DINO/SALAD descriptor
逻辑：query-before-add + L2 retrieval + temporal exclusion
输出：LoopCandidate
参考：VGGT-SLAM 2.0 ImageRetrieval
```

### VGGT-SLAM 2.0

```text
DINO-SALAD checkpoint: dino_salad.ckpt
transform: resize + ImageNet normalize
per-submap descriptors: get_all_submap_embeddings
retrieval: query each frame descriptor against historical submaps
filter: skip current / recent / loop submaps; L2 threshold
output: LoopMatch(query_submap, query_frame, detected_submap, detected_frame)
```

### MASt3R-SLAM

```text
feature source: MASt3R frame.feat
prep_features: prewhiten → projector → attention → postwhiten → top-k local features
index: ASMK / IVF
update: query-before-add
output: top-k historical keyframe indices
```

### ARTDECO

```text
retrieval_database = load_retriever(...)
current keyframe feature stored in self.embeddings
retrieval_database.update(..., add_after_query=True)
retrieval candidates + consecutive keyframes → factor_graph.add_factors
```

### VGGT-SLAM++

```text
retrieval unit: DEM / BEV / tile, not raw image
embedding: DINOv2 tile descriptor
search space: covisibility window / spatial neighbors
output: tile/submap candidate for local correction
```

### M³ / Pi3X

```text
retrieval output: historical keyframe set, not single pair
query frames + history frames form multi-view input
retrieval also participates in keyframe selection
```

## 7.2 Verification：候选验证

### 当前实现：PoseDistanceVerifier

```text
输入：LoopCandidate + trajectory
计算：translation_error, rotation_angle
输出：VerifiedLoop(relative_pose)
用途：baseline / CI / trajectory loop
```

### 当前实现：ModelPairReprojectionVerifier

```text
输入：LoopCandidate + frames + current adapter
构造：[match_frame, query_frame]
调用：adapter.infer(pair_chunk)
读取：T_model_cam, points_model, intrinsics, confidence, valid_mask
验证：query points → match camera projection → match image pixel → match points consistency
输出：VerifiedLoop(relative_pose, relative_h, inlier stats)
```

这是 VGGT-SLAM 2.0 pair inference verifier 的通用化：前端用什么 adapter，pair verifier 就复用什么 adapter，而不是写死 VGGT。

### VGGT-SLAM 2.0

```text
candidate: SALAD LoopMatch
pair input: query frame + retrieved frame
forward: model(lc_frames, compute_similarity=True)
verify: image_match_ratio threshold
accepted output:
  extrinsic_lc
  intrinsic_lc
  depth_lc
  depth_conf_lc
```

### MASt3R-SLAM

```text
candidate: retrieval keyframe pair
forward: mast3r_match_symmetric(model, feat_i, pos_i, feat_j, pos_j, shapes)
confidence:
  Qj = sqrt(Qii[idx_i2j] * Qji)
  Qi = sqrt(Qjj[idx_j2i] * Qij)
filter:
  Q > Q_conf
  valid_match_i / valid_match_j
  bidirectional match_frac > min_match_frac
output:
  idx_i2j / idx_jj2i
  valid masks
  Q weights
```

### ARTDECO

```text
current keyframe vs last keyframe: mast3r_match_asymmetric
retrieval candidates: factor_graph.add_factors
verification: inherited MASt3R dense matching / min_match_frac
mapper receives loop_keyframe_index and densePoint
```

### VGGT-SLAM++

```text
verify unit: DEM tile / spatial neighbor
criteria:
  tile overlap
  elevation consistency
  orthophoto / BEV similarity
  covisibility graph consistency
output:
  local correction request / DEM graph edge
```

### M³ / Pi3X

```text
verify unit: query frames + historical keyframe set
model: Pi3X / M³ multi-view inference
extra head: dense matching head
criteria:
  dense correspondence confidence
  dynamic area suppression
  cross-inference intrinsic alignment
output:
  multiple pose/dense factors + keyframe selection decision
```

## 7.3 Factor：转成后端约束

### 当前实现：LoopFactorAdapter

```text
SE3 / SensorFusion:
  type="loop"
  frame_i = match_frame_id
  frame_j = query_frame_id
  relative_pose

Sim3:
  当前为 overlap_pose anchor
  chunk_id = query_chunk_id
  frame_id = query_frame_id
  T_world_cam = T_world_query

SL4:
  type="sl4_between"
  chunk_i = match_chunk_id
  chunk_j = query_chunk_id
  relative_h
```

### VGGT-SLAM 2.0

```text
node: submap SL4 homography
factor: BetweenFactorSL4(key1, key2, SL4(relative_h), noise)
optimizer: GTSAM LevenbergMarquardtOptimizer
```

### MASt3R-SLAM / ARTDECO

```text
factor is not simple pose edge；
edge data includes dense match indices, valid masks, Q weights；
backend solves with gauss_newton_rays or gauss_newton_calib；
optimized poses feed mapper / Gaussian mapper。
```

### VGGT-SLAM++

```text
factor/correction is DEM graph / local BA / spatial correction；
not necessarily one SE3 edge；
DEM tile acts as compact map node。
```

### M³ / Pi3X

```text
factor construction comes from multi-view inference output；
can produce multiple pose/dense constraints；
Gaussian mapper may jointly optimize camera poses and Gaussians。
```

---

# 8. Backend Node：SE3 / Sim3 / SL4 / Dense GN / SensorFusion

## 8.1 当前 backend family

```text
NoOptBackend
SE3PoseGraphBackend
Sim3PoseGraphBackend
SL4SubmapGraphBackend
SensorFusionGraphBackend
```

## 8.2 SE3 backend

适合：

```text
metric pose graph
有可靠尺度
GNSS/IMU/pair verifier 输出 relative_pose
```

factor：

```text
PriorFactorPose3 / BetweenFactorPose3 或 fallback translation solver
```

适合方法：

```text
MapAnything metric outputs
Pi3X aligned metric outputs
PnP / pose verifier / sensor fusion
MASt3R-Fusion style metric SE3 graph
```

## 8.3 Sim3 backend

适合：

```text
monocular 尺度漂移
chunk-level metric_local alignment
Pi3 / Pi3X / MapAnything local-to-world
feed-forward visual Sim3 constraints
```

当前实现：

```text
world_translation
world_rotation
overlap_pose
chunk-level anchor refinement
```

后续目标：

```text
true sim3_between factor
Umeyama / RANSAC point overlap factor
joint Sim3 graph
visual Sim3 constraints into metric SE3 sensor fusion graph
```

## 8.4 SL4 backend

适合：

```text
VGGT / VGGT-Omega
projective ambiguity
uncalibrated / weakly calibrated submap graph
```

参考 VGGT-SLAM / VGGT-SLAM 2.0：

```text
gtsam.SL4
PriorFactorSL4
BetweenFactorSL4
LevenbergMarquardtOptimizer
```

## 8.5 Dense GN backend

适合：

```text
MASt3R-SLAM / ARTDECO
pointmap + dense matching
```

factor data：

```text
ii / jj
idx_i2j / idx_j2i
valid_match
Q weights
X_canon
confidence
pose variables
```

优化：

```text
gauss_newton_rays: uncalibrated / ray mode
gauss_newton_calib: calibrated K mode
generic central camera / ray interface: support time-varying camera, zoom camera, unknown intrinsics
```

## 8.6 SensorFusion backend

适合：

```text
GNSS / IMU / wheel / pre-integrated relative pose
MASt3R-Fusion style visual + IMU + GNSS fusion
```

当前优先支持 pseudo factors：

```text
world_translation
world_rotation
sensor_relative_pose
```

后续 raw IMU：

```text
velocity
bias
IMU preintegration buffer
time synchronization
sliding-window optimization
global optimization with loop closures
```

---

# 9. Mapping Node：PointCloud / DEM / Gaussian

## 9.1 当前实现：PointCloudMap

```text
输入：WorldChunkPrediction.points_world
去重：overlap frame 不重复融合
输出：points / map_summary
```

适合：

```text
调试几何一致性
导出 PLY/NPY
验证前端/后端是否跑通
```

## 9.2 DEMMap：VGGT-SLAM++ / UAV / DOM

后续设计：

```text
points_world / depth → local DEM tile
DEM tile → DINOv2 embedding
DEM tile → covisibility graph
tile overlap → local correction
DEM / DOM export
```

技术节点：

```text
TileLoopRetrieval
TileOverlapVerifier
MappingCorrectionRequest
DEM backend residual
spatially corrective backend
high-cadence local BA
```

## 9.3 GaussianMap：ARTDECO / M³ / SplaTAM

ARTDECO / M³ / SplaTAM 的共同启发是：地图可以不是普通点云，而是可渲染的 Gaussian representation。

```text
ARTDECO:
  3D foundation model pose/point prediction
  → Gaussian decoder
  → structured / hierarchical Gaussians
  → LoD-aware rendering

M³:
  multi-view foundation model + dense matching head
  → monocular Gaussian Splatting SLAM
  → camera pose + Gaussian map joint optimization

SplaTAM:
  explicit Gaussian representation
  → tracking / mapping / rendering
```

映射：

```text
GaussianMap.integrate(world_pred, features)
GaussianMap.optimize_pose_map(loop_correction)
MappingCorrectionRequest
rendering residual / photometric residual
pose-map joint optimization
```

---

# 10. 方法到代码抽象的完整映射

## 10.1 DUSt3R / MASt3R base model

```text
Data:
  pointmap / confidence / matching features
Adapter:
  outputs points_model + matching_features
Verification:
  pointmap matching / reciprocal matching
Backend:
  global alignment or dense GN
```

本项目对应：

```text
StandardChunkPrediction.points_model
StandardChunkPrediction.matching_features
PointCloudOverlapVerifier / Mast3RDensePairVerifier
DenseGNBackend
```

## 10.2 VGGT-SLAM 2.0

```text
Frame/Chunk:
  Chunk == submap
Adapter:
  VGGTAdapter
  pair verifier uses model(lc_frames, compute_similarity=True)
Retrieval:
  DINO/SALAD ImageRetrieval
  per-frame descriptor inside submap
  L2 retrieval against historical submaps
Verification:
  VGGT pair inference
  image_match_ratio threshold
  outputs loop pose/depth/conf
Factor:
  SL4 BetweenFactor
Backend:
  GTSAM SL4 graph
Mapping:
  dense submap points / visualization
```

本项目对应：

```text
DinoSaladEmbeddingProvider
ModelPairReprojectionVerifier 或未来 VGGTImageMatchVerifier
LoopFactorAdapter(backend_type="sl4_submap_graph")
SL4SubmapGraphBackend
```

## 10.3 MASt3R-SLAM

```text
Frame/Chunk:
  KeyFrame + pointmap + retrieval feature
Adapter:
  MASt3RAdapter should output points_model, confidence, matching_features
Retrieval:
  MASt3R RetrievalDatabase
  ASMK / IVF
  query-before-add
Verification:
  mast3r_match_symmetric
  Q confidence
  bidirectional match fraction
Factor:
  dense matching factor data
Backend:
  custom Gauss-Newton rays/calib
Mapping:
  pointmap-based dense map
```

本项目后续对应：

```text
Mast3RRetrievalProvider
Mast3RDensePairVerifier
DenseGNBackend
generic ray/camera interface
```

## 10.4 MASt3R-Fusion

```text
Adapter:
  feed-forward pointmap regression
Visual factor:
  Sim(3)-based visual alignment constraints
Sensor factor:
  IMU / GNSS / wheel / relative pose
Backend:
  metric-scale SE(3) factor graph
Optimization:
  realtime sliding window + global loop optimization
```

本项目后续对应：

```text
Sim3 visual factor
SensorFusionGraphBackend
raw IMU preintegration
sliding-window backend mode
global optimization with loop closures
```

## 10.5 ARTDECO

```text
Frame/Chunk:
  keyframe from frontend
Adapter:
  MASt3R/Pi3-like model for pose and point prediction
Retrieval:
  retrieval_database.update
Verification:
  MASt3R asymmetric/symmetric matching
Factor:
  FactorGraph dense factors
Backend:
  GN calib/rays
Mapping:
  structured hierarchical Gaussian mapper
  loop_keyframe_index as metadata
```

本项目后续对应：

```text
DenseGNBackend
GaussianMap
loop metadata passed to mapper
```

## 10.6 VGGT-SLAM++

```text
Frame/Chunk:
  VGGT submap
Adapter:
  VGGT feed-forward geometry
Retrieval:
  DEM tile DINOv2 embedding
  covisibility-window VPR
Verification:
  tile overlap / elevation / spatial consistency
Factor:
  DEM graph / local correction
Backend:
  spatially corrective backend + high-cadence LBA
Mapping:
  compact DEM tiles
```

本项目后续对应：

```text
DEMMap
TileLoopRetrieval
TileOverlapVerifier
MappingCorrectionRequest
```

## 10.7 M³ / Pi3X

```text
Frame/Chunk:
  query frames + retrieved history frames
Adapter:
  Pi3X / M³ multi-view model
Retrieval:
  historical keyframe set retrieval
Verification:
  multi-view inference
  dense matching head
  dynamic area suppression
  cross-inference intrinsic alignment
Factor:
  multiple pose/dense factors
Backend:
  pose graph + Gaussian pose-map optimization
Mapping:
  Monocular Gaussian Splatting SLAM
```

本项目后续对应：

```text
MultiFrameLoopCandidate
MultiViewModelVerifier
Pi3X retrieved-keyframe wrapper
GaussianMap correction
```

## 10.8 MapAnything / G-CUT3R

```text
Data:
  images + intrinsics + poses + depth + partial reconstruction
Adapter:
  geometry-prior-capable model input
Prior:
  intrinsics / rays / pose / translation / rotation / depth / partial points
Output:
  metric 3D scene geometry and cameras
```

本项目对应：

```text
ModelSpec prior capability flags
PriorBuilder
MapAnythingAdapter / Pi3XAdapter
BackendPrior
CoordinateNormalizer
```

## 10.9 CUT3R / Spann3R

```text
Data:
  streaming frames
Adapter:
  persistent state / spatial memory
Output:
  common-coordinate or global-coordinate pointmaps
Backend:
  may need less global alignment but still benefits from loop / sensor priors
```

本项目后续对应：

```text
ModelSpec.is_stateful
adapter memory_state
stateful GenericFeedForwardSLAM mode
```

---

# 11. 当前已实现与后续路线

## 11.1 当前已实现

```text
GenericFeedForwardSLAM 主流程
Fixed ChunkManager
PriorBuilder
CoordinateNormalizer
OverlapManager
PointCloudMap
SE3 / Sim3 / SL4 / SensorFusion backend shell
TemporalDistanceLoopRetrieval
DinoSaladEmbeddingProvider
PoseDistanceVerifier
ModelPairReprojectionVerifier
LoopFactorAdapter
T600 exporter
```

## 11.2 下一步建议

### P0：稳定当前通用 pipeline

```text
1. 运行所有 loop tests；
2. 用真实 VGGT / Pi3 / Pi3X / MapAnything 验证 adapter 输出字段；
3. 确认 ModelPairReprojectionVerifier 在真实 dense points 上阈值是否合理。
```

### P1：PointCloudOverlapVerifier

原因：

```text
所有 dense feed-forward 模型都能受益；
比 pose distance 更强；
比模型专用 verifier 更通用。
```

### P2：VGGT-SLAM 2.0 专用增强

```text
DinoSaladEmbeddingProvider + VGGT pair inference diagnostics
如果 adapter 暴露 image_match_ratio，则加入 threshold；
SL4SubmapGraphBackend 真实 gtsam.SL4 验证。
```

### P3：MASt3R / ARTDECO 路线

```text
Mast3RRetrievalProvider
Mast3RDensePairVerifier
DenseGNBackend
GaussianMap metadata interface
```

### P4：VGGT-SLAM++ / UAV DOM 路线

```text
DEMMap
DEM tile descriptor
TileLoopRetrieval
TileOverlapVerifier
DEM local correction backend
```

### P5：M³ / Pi3X multi-view 路线

```text
MultiFrameLoopCandidate
MultiViewModelVerifier
retrieved historical keyframes + current window
Gaussian pose-map correction
```

### P6：Stateful model 路线

```text
ModelSpec.is_stateful
adapter.reset_state / memory_state
streaming inference runner
CUT3R / Spann3R style persistent memory validation
```

---

# 12. 关键边界

```text
1. GenericFeedForwardSLAM 不应该知道 VGGT / MASt3R / Pi3X 的内部细节；
2. Retrieval 只产生候选，不能直接加 factor；
3. Verification 必须输出可解释的 score / inlier_ratio / metadata；
4. FactorAdapter 根据 backend_type 分流；
5. Mapping correction 不应和 pose graph factor 混在一起；
6. 对 projective 模型优先 SL4；
7. 对 metric local 模型优先 Sim3；
8. 对 sensor prior 强的场景可接 SE3/SensorFusion；
9. 对 dense matching 模型可接 DenseGNBackend；
10. 对 UAV/DOM 最终应转向 DEM/Tile abstraction；
11. 对 Gaussian SLAM 最终应通过 MappingCorrectionRequest 接入，而不是塞进 loop factor；
12. 对 stateful model，应让 adapter 管理 memory，主流程只调用统一接口。
```
