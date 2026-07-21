import warnings
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from mapanything.models.mapanything import MapAnything

from mapanything.utils.inference import (
    postprocess_model_outputs_for_inference,
    preprocess_input_views_for_inference,
    validate_input_views_for_inference,
)
from uniception.models.info_sharing.base import MultiViewTransformerInput
from uniception.models.prediction_heads.base import PredictionHeadLayeredInput
from uniception.models.prediction_heads.dpt import (
    DPTFeature,
    DPTSegmentationProcessor,
)


class MapAnythingSemantic(MapAnything):
    """
    MapAnything + Semantic Segmentation head.

    说明：
    - 保留原 MapAnything 所有几何预测分支（pointmap / raymap / pose / conf / mask 等）
    - 新增 semantic 分支（DPT feature head + DPTSegmentationProcessor）
    - forward / infer 都同时返回几何结果 + 语义结果
    - 输入格式与原 MapAnything 保持一致（views 不变）

    注意：
    - 当前实现为了“最小侵入、最大兼容”，几何分支和语义分支在 forward 中分别跑：
        geometry: super().forward(...)
        semantic: _forward_semantic_only(...)
      即会重复计算 trunk（编码+融合+多视角共享）一次。
    - 优点：改动小、稳定、不破坏原几何逻辑
    - 后续如果你要优化速度，我可以再给你“单次共享trunk”的版本。
    """

    def __init__(
        self,
        *args,
        semantic_num_classes: int,
        semantic_ignore_index: int = 0,
        return_semantic_probs: bool = False,
        **kwargs,
    ):
        # 先存属性（因为父类 __init__ 会调用被 override 的方法）
        self.semantic_num_classes = int(semantic_num_classes)
        self.semantic_ignore_index = int(semantic_ignore_index)
        self.return_semantic_probs = bool(return_semantic_probs)

        super().__init__(*args, **kwargs)

        # 记录到可序列化参数中（如果父类使用 class_init_args）
        if hasattr(self, "class_init_args") and isinstance(self.class_init_args, dict):
            self.class_init_args.update(
                {
                    "semantic_num_classes": self.semantic_num_classes,
                    "semantic_ignore_index": self.semantic_ignore_index,
                    "return_semantic_probs": self.return_semantic_probs,
                }
            )

    # -------------------------------------------------------------------------
    # 初始化：保留原几何头 + 新增 semantic 头
    # -------------------------------------------------------------------------
    def _initialize_prediction_heads(self, pred_head_config):
        """
        在原有几何预测头基础上，新增 semantic head。
        """
        # 先走原 MapAnything 的初始化（几何头、pose头、scale头等）
        super()._initialize_prediction_heads(pred_head_config)

        # 语义头当前依赖 DPT 风格的 layered features
        if self.pred_head_type not in ["dpt", "dpt+pose"]:
            raise ValueError(
                f"MapAnythingSemantic currently supports pred_head_type in ['dpt', 'dpt+pose'], "
                f"but got {self.pred_head_type}"
            )

        # 单独再建一个 semantic DPT feature head（不覆盖几何 dense_head）
        # 注意：父类 _initialize_prediction_heads 已经把 pred_head_config["feature_head"] 里的依赖补齐了
        self.semantic_dpt_feature_head = DPTFeature(**pred_head_config["feature_head"])

        # 使用 UniCeption 原生 DPTSegmentationProcessor
        # 支持通过 pred_head_config 可选传入配置：
        #   pred_head_config["semantic_processor_head"] 或 pred_head_config["semantic_segmentation_processor"]
        # 否则使用默认参数
        semantic_processor_cfg = {}
        if "semantic_processor_head" in pred_head_config:
            semantic_processor_cfg = dict(pred_head_config["semantic_processor_head"])
        elif "semantic_segmentation_processor" in pred_head_config:
            semantic_processor_cfg = dict(pred_head_config["semantic_segmentation_processor"])

        semantic_feature_dim = pred_head_config["feature_head"]["feature_dim"]

        # 强制对齐 input/output 维度（用户配置可覆盖 hidden_dim/use_bn 等）
        semantic_processor_cfg["input_feature_dim"] = semantic_feature_dim
        semantic_processor_cfg["output_dim"] = self.semantic_num_classes

        self.semantic_seg_processor = DPTSegmentationProcessor(**semantic_processor_cfg)

    def _load_pretrained_weights(self):
        """
        重载为 strict=False，允许加载原 MapAnything ckpt（缺少 semantic head 权重）。
        """
        if self.pretrained_checkpoint_path is None:
            return

        if not self.load_specific_pretrained_submodules:
            print(
                f"Loading pretrained MapAnything/MapAnythingSemantic weights from {self.pretrained_checkpoint_path} ..."
            )
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            msg = self.load_state_dict(ckpt["model"], strict=False)
            print(msg)
        else:
            print(
                f"Loading pretrained weights from {self.pretrained_checkpoint_path} "
                f"for specific submodules: {self.specific_pretrained_submodules} ..."
            )
            assert self.specific_pretrained_submodules is not None, (
                "specific_pretrained_submodules cannot be None when "
                "load_specific_pretrained_submodules=True."
            )
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            filtered_ckpt = {}
            for ckpt_key, ckpt_value in ckpt["model"].items():
                for submodule in self.specific_pretrained_submodules:
                    if ckpt_key.startswith(submodule):
                        filtered_ckpt[ckpt_key] = ckpt_value
                        break
            msg = self.load_state_dict(filtered_ckpt, strict=False)
            print(msg)

    # -------------------------------------------------------------------------
    # semantic branch helpers
    # -------------------------------------------------------------------------
    def _build_semantic_dense_head_inputs(
        self, views: List[Dict[str, Any]]
    ) -> Tuple[List[torch.Tensor], Tuple[int, int], int]:
        """
        复用原 forward 前半段逻辑，构建 semantic DPT 所需的 layered features。

        Returns:
            dense_head_inputs_list: DPT layered features list (len=4), each shape (B*V, C, h, w)
            img_shape: (H, W)
            num_views: V
        """
        batch_size_per_view, _, height, width = views[0]["img"].shape
        img_shape = (int(height), int(width))
        num_views = len(views)

        # 1) image encoder
        all_encoder_features_across_views, all_encoder_registers_across_views = (
            self._encode_n_views(views)
        )

        # 2) geometric fusion（高精度防止数值问题）
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            all_encoder_features_across_views = (
                self._encode_and_fuse_optional_geometric_inputs(
                    views, all_encoder_features_across_views
                )
            )

        # 3) info sharing transformer（沿用 scale_token；semantic head 本身不使用 scale 输出）
        input_scale_token = (
            self.scale_token.unsqueeze(0).unsqueeze(-1).repeat(batch_size_per_view, 1, 1)
        )  # (B, C, 1)

        info_sharing_input = MultiViewTransformerInput(
            features=all_encoder_features_across_views,
            additional_input_tokens_per_view=all_encoder_registers_across_views,
            additional_input_tokens=input_scale_token,
        )

        final_info_sharing_multi_view_feat = None
        intermediate_info_sharing_multi_view_feat = None
        if self.info_sharing_return_type == "no_intermediate_features":
            # 语义头依赖 layered features（DPT），这里一般不建议 no_intermediate_features
            final_info_sharing_multi_view_feat = self.info_sharing(info_sharing_input)
            raise ValueError(
                "MapAnythingSemantic semantic head expects info_sharing_return_type='intermediate_features' "
                "to build DPT layered inputs."
            )
        elif self.info_sharing_return_type == "intermediate_features":
            (
                final_info_sharing_multi_view_feat,
                intermediate_info_sharing_multi_view_feat,
            ) = self.info_sharing(info_sharing_input)
        else:
            raise ValueError(
                f"Invalid info_sharing_return_type: {self.info_sharing_return_type}"
            )

        # 4) 组装 DPT layered inputs（与原类 forward 中 DPT 分支一致）
        dense_head_inputs_list = []
        if self.use_encoder_features_for_dpt:
            # encoder feat
            stacked_encoder_features = torch.cat(all_encoder_features_across_views, dim=0)
            dense_head_inputs_list.append(stacked_encoder_features)

            # interm 1
            stacked_intermediate_features_1 = torch.cat(
                intermediate_info_sharing_multi_view_feat[0].features, dim=0
            )
            dense_head_inputs_list.append(stacked_intermediate_features_1)

            # interm 2
            stacked_intermediate_features_2 = torch.cat(
                intermediate_info_sharing_multi_view_feat[1].features, dim=0
            )
            dense_head_inputs_list.append(stacked_intermediate_features_2)

            # final
            stacked_final_features = torch.cat(
                final_info_sharing_multi_view_feat.features, dim=0
            )
            dense_head_inputs_list.append(stacked_final_features)
        else:
            # interm 1
            stacked_intermediate_features_1 = torch.cat(
                intermediate_info_sharing_multi_view_feat[0].features, dim=0
            )
            dense_head_inputs_list.append(stacked_intermediate_features_1)

            # interm 2
            stacked_intermediate_features_2 = torch.cat(
                intermediate_info_sharing_multi_view_feat[1].features, dim=0
            )
            dense_head_inputs_list.append(stacked_intermediate_features_2)

            # interm 3
            stacked_intermediate_features_3 = torch.cat(
                intermediate_info_sharing_multi_view_feat[2].features, dim=0
            )
            dense_head_inputs_list.append(stacked_intermediate_features_3)

            # final
            stacked_final_features = torch.cat(
                final_info_sharing_multi_view_feat.features, dim=0
            )
            dense_head_inputs_list.append(stacked_final_features)

        return dense_head_inputs_list, img_shape, num_views

    def _forward_semantic_only(
        self,
        views: List[Dict[str, Any]],
        memory_efficient_inference: bool = False,
        minibatch_size: int = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        仅跑 semantic 分支，返回每个 view 的 semantic 输出（不含几何输出）。
        """
        dense_head_inputs_list, img_shape, num_views = self._build_semantic_dense_head_inputs(
            views
        )

        device_type = "cuda" if self.device.type == "cuda" else "cpu"

        # memory efficient 分块推理（针对 semantic head）
        def _run_semantic_minibatch(dense_inputs_list, out_hw):
            sem_feat_out = self.semantic_dpt_feature_head(
                PredictionHeadLayeredInput(
                    list_features=dense_inputs_list,
                    target_output_shape=out_hw,
                )
            )
            # 使用 DPTSegmentationProcessor 输出 PixelTaskOutput(decoded_channels=...)
            sem_logits_out = self.semantic_seg_processor(sem_feat_out)
            sem_logits = sem_logits_out.decoded_channels
            return sem_logits

        with torch.autocast(device_type=device_type, enabled=False):
            if not memory_efficient_inference:
                semantic_logits = _run_semantic_minibatch(dense_head_inputs_list, img_shape)
            else:
                batch_size = dense_head_inputs_list[0].shape[0]
                if minibatch_size is not None:
                    mb = int(minibatch_size)
                else:
                    mb = self._compute_adaptive_minibatch_size()
                mb = max(mb, 1)

                num_batches = (batch_size + mb - 1) // mb
                logits_list = []
                for bi in range(num_batches):
                    s = bi * mb
                    e = min((bi + 1) * mb, batch_size)
                    mini_inputs = [x[s:e] for x in dense_head_inputs_list]
                    logits_mini = _run_semantic_minibatch(mini_inputs, img_shape)
                    logits_list.append(logits_mini)
                semantic_logits = torch.cat(logits_list, dim=0)

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        # 拆回每个 view
        logits_per_view = semantic_logits.chunk(num_views, dim=0)

        sem_res = []
        for logits_i in logits_per_view:
            probs_i = F.softmax(logits_i, dim=1)
            conf_i, pred_i = torch.max(probs_i, dim=1)  # (B,H,W)

            out = {
                "semantic_logits": logits_i,  # (B,C,H,W)
                "semantic_pred": pred_i,      # (B,H,W)
                "semantic_conf": conf_i,      # (B,H,W)
            }
            if self.return_semantic_probs:
                out["semantic_probs"] = probs_i
            sem_res.append(out)

        return sem_res

    @staticmethod
    def _merge_viewwise_outputs(
        geom_res: List[Dict[str, torch.Tensor]],
        sem_res: List[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        按 view 合并几何输出和语义输出。
        """
        assert len(geom_res) == len(sem_res), "geom_res and sem_res must have same number of views"
        merged = []
        for i in range(len(geom_res)):
            out = {}
            out.update(geom_res[i])  # 原几何字段
            out.update(sem_res[i])   # 新增 semantic 字段（命名避免冲突）
            merged.append(out)
        return merged

    # -------------------------------------------------------------------------
    # forward: 同时返回几何 + semantic
    # -------------------------------------------------------------------------
    def forward(self, views, memory_efficient_inference=False, minibatch_size=None):
        """
        返回：
        - 原 MapAnything 几何输出（pts3d/depth/ray/cam/conf/mask...）
        - 额外 semantic 输出：
            semantic_logits: (B,C,H,W)
            semantic_pred:   (B,H,W)
            semantic_conf:   (B,H,W)
            semantic_probs:  (B,C,H,W) [可选]
        """
        # 1) 原几何输出（完全复用父类逻辑，保证兼容）
        geom_res = super().forward(
            views,
            memory_efficient_inference=memory_efficient_inference,
            minibatch_size=minibatch_size,
        )

        # 2) 语义输出（新增分支）
        sem_res = self._forward_semantic_only(
            views,
            memory_efficient_inference=memory_efficient_inference,
            minibatch_size=minibatch_size,
        )

        # 3) 合并输出
        return self._merge_viewwise_outputs(geom_res, sem_res)

    # -------------------------------------------------------------------------
    # infer: 同时返回“几何后处理结果 + semantic结果”
    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def infer(
        self,
        views: List[Dict[str, Any]],
        memory_efficient_inference: bool = True,
        minibatch_size: int = None,
        use_amp: bool = True,
        amp_dtype: str = "bf16",
        apply_mask: bool = True,
        mask_edges: bool = True,
        edge_normal_threshold: float = 5.0,
        edge_depth_threshold: float = 0.03,
        apply_confidence_mask: bool = False,
        confidence_percentile: float = 10,
        ignore_calibration_inputs: bool = False,
        ignore_depth_inputs: bool = False,
        ignore_pose_inputs: bool = False,
        ignore_depth_scale_inputs: bool = False,
        ignore_pose_scale_inputs: bool = False,
        use_multiview_confidence: bool = False,
        multiview_conf_depth_abs_thresh: float = 0.02,
        multiview_conf_depth_rel_thresh: float = 0.02,
        apply_semantic_mask: bool = True,  # 新增：是否对语义输出应用 mask
    ) -> List[Dict[str, torch.Tensor]]:
        """
        同时返回：
        1) 原几何 infer 后处理结果（与 MapAnything.infer 一致）
        2) 语义输出（semantic_logits / semantic_pred / semantic_conf）
        """
        # AMP dtype
        if use_amp:
            if amp_dtype == "fp16":
                amp_dtype = torch.float16
            elif amp_dtype == "bf16":
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    amp_dtype = torch.bfloat16
                else:
                    warnings.warn("bf16 is not supported on this device. Using fp16 instead.")
                    amp_dtype = torch.float16
            elif amp_dtype == "fp32":
                amp_dtype = torch.float32
            else:
                raise ValueError(f"Invalid amp_dtype: {amp_dtype}")
        else:
            amp_dtype = torch.float32

        # Validate
        validated_views = validate_input_views_for_inference(views)

        # Move to model device
        ignore_keys = {"instance", "idx", "true_shape", "data_norm_type"}
        for view in validated_views:
            for name in list(view.keys()):
                if name in ignore_keys:
                    continue
                val = view[name]
                if name == "camera_poses" and isinstance(val, tuple):
                    view[name] = tuple(x.to(self.device, non_blocking=True) for x in val)
                elif hasattr(val, "to"):
                    view[name] = val.to(self.device, non_blocking=True)

        # Preprocess
        processed_views = preprocess_input_views_for_inference(validated_views)

        # 配置几何输入开关（与原 infer 一致）
        self._configure_geometric_input_config(
            use_calibration=not ignore_calibration_inputs,
            use_depth=not ignore_depth_inputs,
            use_pose=not ignore_pose_inputs,
            use_depth_scale=not ignore_depth_scale_inputs,
            use_pose_scale=not ignore_pose_scale_inputs,
        )

        device_type = "cuda" if self.device.type == "cuda" else "cpu"

        try:
            # -------------------------
            # 1) 几何 raw forward（只跑父类 forward）
            # -------------------------
            with torch.autocast(device_type=device_type, enabled=bool(use_amp), dtype=amp_dtype):
                geom_raw_preds = super().forward(
                    processed_views,
                    memory_efficient_inference=memory_efficient_inference,
                    minibatch_size=minibatch_size,
                )

            # 几何后处理（沿用原逻辑）
            geom_preds = postprocess_model_outputs_for_inference(
                raw_outputs=geom_raw_preds,
                input_views=processed_views,
                apply_mask=apply_mask,
                mask_edges=mask_edges,
                edge_normal_threshold=edge_normal_threshold,
                edge_depth_threshold=edge_depth_threshold,
                apply_confidence_mask=apply_confidence_mask,
                confidence_percentile=confidence_percentile,
                use_multiview_confidence=use_multiview_confidence,
                multiview_conf_depth_abs_thresh=multiview_conf_depth_abs_thresh,
                multiview_conf_depth_rel_thresh=multiview_conf_depth_rel_thresh,
            )

            # -------------------------
            # 2) semantic 分支
            # -------------------------
            with torch.autocast(device_type=device_type, enabled=bool(use_amp), dtype=amp_dtype):
                sem_preds = self._forward_semantic_only(
                    processed_views,
                    memory_efficient_inference=memory_efficient_inference,
                    minibatch_size=minibatch_size,
                )

            # 可选：对 semantic 输出应用 mask（优先用几何后处理的mask）
            if apply_semantic_mask:
                for i in range(len(sem_preds)):
                    mask = None

                    # 优先使用几何后处理后的总mask
                    if "mask" in geom_preds[i]:
                        mask = geom_preds[i]["mask"]
                    # 次选 non_ambiguous_mask
                    elif "non_ambiguous_mask" in geom_preds[i]:
                        mask = geom_preds[i]["non_ambiguous_mask"]
                    elif "non_ambiguous_mask" in processed_views[i]:
                        mask = processed_views[i]["non_ambiguous_mask"]
                    elif "mask" in processed_views[i]:
                        mask = processed_views[i]["mask"]

                    if mask is None:
                        continue

                    # 统一成 (B,H,W) bool
                    if isinstance(mask, torch.Tensor):
                        mask_t = mask.to(sem_preds[i]["semantic_pred"].device)
                    else:
                        mask_t = torch.as_tensor(mask, device=sem_preds[i]["semantic_pred"].device)

                    if mask_t.dim() == 4 and mask_t.shape[-1] == 1:
                        mask_t = mask_t[..., 0]
                    elif mask_t.dim() == 4 and mask_t.shape[1] == 1:
                        # 兼容 (B,1,H,W)
                        mask_t = mask_t[:, 0]
                    mask_t = mask_t.bool()

                    pred = sem_preds[i]["semantic_pred"]
                    conf = sem_preds[i]["semantic_conf"]

                    # 尺寸不一致时最近邻 resize
                    if mask_t.shape != pred.shape:
                        mask_t = F.interpolate(
                            mask_t.unsqueeze(1).float(),
                            size=pred.shape[-2:],
                            mode="nearest",
                        ).squeeze(1).bool()

                    pred = pred.clone()
                    conf = conf.clone()
                    pred[~mask_t] = self.semantic_ignore_index
                    conf[~mask_t] = 0.0

                    sem_preds[i]["semantic_pred"] = pred
                    sem_preds[i]["semantic_conf"] = conf
                    sem_preds[i]["semantic_valid_mask"] = mask_t

            # 合并几何+语义输出
            merged_preds = self._merge_viewwise_outputs(geom_preds, sem_preds)
            return merged_preds

        finally:
            # 防止异常中断导致配置污染
            self._restore_original_geometric_input_config()
            