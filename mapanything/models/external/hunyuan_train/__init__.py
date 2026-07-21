import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from hunyuanworld_mirror.models.models.worldmirror import WorldMirror


class HunyuanWrapper(nn.Module):
    """
    Trainable HunyuanWorld-Mirror wrapper for the evaluate_forward_models framework.

    This adapter keeps the model-side semantics close to the original Hunyuan training code:
    - inputs are passed as a batched dict with keys like img/camera_poses/camera_intrs/depthmap
    - cond_flags follow the original training wrapper convention:
        [camera_pose, depthmap, camera_intrs]
    - outputs are unpacked into the per-view list format expected by MapAnything's training loop.
    """

    def __init__(
        self,
        name,
        torch_hub_force_reload,
        hf_model_name: str = "tencent/HunyuanWorld-Mirror",
        geometric_input_config: Optional[dict] = None,
        load_pretrained_weights: bool = True,
        use_conditioning: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.use_conditioning = use_conditioning
        self.geometric_input_config = geometric_input_config or {}

        if load_pretrained_weights:
            if torch_hub_force_reload:
                self.model = WorldMirror.from_pretrained(
                    hf_model_name,
                    force_download=True,
                    **kwargs,
                )
            else:
                self.model = WorldMirror.from_pretrained(
                    hf_model_name,
                    **kwargs,
                )
        else:
            self.model = WorldMirror(**kwargs)

    def _stack_inputs(self, views: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        device = views[0]["img"].device
        inputs = {
            "img": torch.stack([v["img"] for v in views], dim=1),
        }

        if all("camera_pose" in v for v in views):
            inputs["camera_poses"] = torch.stack(
                [v["camera_pose"].to(device) for v in views], dim=1
            )
        if all("camera_intrinsics" in v for v in views):
            inputs["camera_intrs"] = torch.stack(
                [v["camera_intrinsics"].to(device) for v in views], dim=1
            )
        if all("depthmap" in v for v in views):
            inputs["depthmap"] = torch.stack(
                [v["depthmap"].to(device) for v in views], dim=1
            )
        if all("valid_mask" in v for v in views):
            inputs["valid_mask"] = torch.stack(
                [v["valid_mask"].to(device) for v in views], dim=1
            )
        if all("pts3d" in v for v in views):
            inputs["pts3d"] = torch.stack(
                [v["pts3d"].to(device) for v in views], dim=1
            )
        if all("normals" in v for v in views):
            inputs["normals"] = torch.stack(
                [v["normals"].to(device) for v in views], dim=1
            )
        if "dataset" in views[0]:
            dataset_field = views[0]["dataset"]
            if isinstance(dataset_field, list):
                # collated string field, already batch-major per view
                inputs["dataset"] = dataset_field
            else:
                # uncollated / fallback
                inputs["dataset"] = [v.get("dataset", "unknown") for v in views]

        return inputs

    def _sample_bool(self, prob: float, device: torch.device) -> bool:
        return bool(torch.rand(1, device=device) < prob)

    def _build_cond_flags(
        self,
        views: List[Dict[str, torch.Tensor]],
        device: torch.device,
        training: bool,
    ) -> List[int]:
        """
        Original Hunyuan training wrapper uses:
            [1,0,0] -> camera prior
            [0,1,0] -> depth prior
            [0,0,1] -> intrinsics prior
        """
        cond_flags = [0, 0, 0]

        if not self.use_conditioning:
            return cond_flags

        cfg = self.geometric_input_config or {}
        overall_prob = float(cfg.get("overall_prob", 1.0))
        cam_prob = float(cfg.get("cam_prob", 0.0))
        depth_prob = float(cfg.get("depth_prob", 0.0))
        intr_prob = float(cfg.get("ray_dirs_prob", 0.0))

        if training:
            if not self._sample_bool(overall_prob, device):
                return cond_flags
            use_pose = cam_prob > 0 and all("camera_pose" in v for v in views) and self._sample_bool(cam_prob, device)
            use_depth = depth_prob > 0 and all("depthmap" in v for v in views) and self._sample_bool(depth_prob, device)
            use_intr = intr_prob > 0 and all("camera_intrinsics" in v for v in views) and self._sample_bool(intr_prob, device)
        else:
            use_pose = cam_prob > 0 and all("camera_pose" in v for v in views)
            use_depth = depth_prob > 0 and all("depthmap" in v for v in views)
            use_intr = intr_prob > 0 and all("camera_intrinsics" in v for v in views)

        if use_pose:
            cond_flags[0] = 1
        if use_depth:
            cond_flags[1] = 1
        if use_intr:
            cond_flags[2] = 1
        return cond_flags

    def forward(self, views: List[Dict[str, torch.Tensor]]):
        device = views[0]["img"].device
        inputs = self._stack_inputs(views)
        cond_flags = self._build_cond_flags(views, device=device, training=self.training)

        preds = self.model(
            inputs,
            cond_flags=cond_flags,
            is_inference=not self.training,
        )

        num_views = len(views)
        res = []
        for i in range(num_views):
            out = {
                "depth_z": preds["depth"][:, i],
                "conf": preds["depth_conf"][:, i],
                "pts3d": preds["pts3d"][:, i],
                "pts3d_conf": preds["pts3d_conf"][:, i],
                "pred_c2w": preds["camera_poses"][:, i],
                "pred_intrinsics": preds["camera_intrs"][:, i],
                "pred_camera_params": preds["camera_params"][:, i],
            }
            if "normals" in preds:
                out["normals"] = preds["normals"][:, i]
                out["normals_conf"] = preds["normals_conf"][:, i]
            res.append(out)
        return res
