# Documentation Index

本文档是 `docs/` 的入口。后续 SLAM 开发只维护下面两个主文档。

## 1. `slam_unified_task_queue.md`

唯一任务队列入口。

用途：

```text
- 查看当前实现状态；
- 决定下一步做哪个任务；
- 记录每个任务修改文件、测试命令、测试状态、遗留问题；
- 标记 DONE / PARTIAL / TODO / DEFERRED / FROZEN；
- 标注参考/复用的开源项目来源。
```

维护规则：

```text
任何实现、修复、测试、重构后，都要同步更新本文档中的任务状态。
```

## 2. `slam_pipeline_technical_nodes.md`

唯一技术设计入口。

用途：

```text
- 解释当前 Generic Feed-forward SLAM 框架；
- 按代码抽象节点说明技术设计；
- 对齐 VGGT-SLAM 2.0、MASt3R-SLAM、ARTDECO、VGGT-SLAM++、M³/Pi3X、MASt3R-Fusion、CUT3R/Spann3R、MapAnything/G-CUT3R 等方法；
- 说明 loop closure 的 retrieval / verification / factor；
- 说明 SE3 / Sim3 / SL4 / Dense GN / SensorFusion backend 的边界；
- 说明 PointCloud / DEM / Gaussian mapping 的扩展方向；
- 说明后续扩展应接到哪个代码节点。
```

维护规则：

```text
新的技术设计、方法对比、接口边界、可复用开源实现，只更新本文档。
```

## Maintenance rule

```text
1. 新任务状态只更新 slam_unified_task_queue.md；
2. 新技术设计只更新 slam_pipeline_technical_nodes.md；
3. 不再新增零散的 loop closure 调研文档；
4. 临时调研先在聊天/issue 中讨论，确认后合并进主技术文档；
5. 旧的重复/临时文档可删除，不需要继续维护。
```
