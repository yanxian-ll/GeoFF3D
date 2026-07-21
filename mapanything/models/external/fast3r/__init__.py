"""MapAnything inference wrapper for Fast3R."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from mapanything.models.external.vggt.utils.rotation import mat_to_quat


def _ensure_fast3r_importable() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    third_party = repo_root / "third_party" / "fast3r"
    if third_party.exists() and str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))


def _resolve_inference_precision(dtype: str):
    key = str(dtype).strip().lower()
    if key in {"float32", "fp32", "32"}:
        return "32"
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if key in {"float16", "fp16", "16"}:
        return "16-mixed"
    raise ValueError(f"Unsupported Fast3R dtype: {dtype}")


def _first_data_norm_type(view: Dict[str, object]) -> str:
    value = view.get("data_norm_type", "identity")
    if isinstance(value, (list, tuple)):
        return str(value[0])
    return str(value)


class Fast3RWrapper(torch.nn.Module):
    """Fast3R exposed through the MapAnything external model contract.

    Fast3R predicts all views in a single forward pass. This wrapper is intended
    for inference/benchmarking and returns the standard per-view dictionaries
    used by the spatial RRD pipeline.
    """

    def __init__(
        self,
        name: str,
        torch_hub_force_reload: bool = False,
        load_pretrained_weights: bool = True,
        pretrained_model_name_or_path: str = "checkpoints/Fast3R_ViT_Large_512",
        dtype: str = "float32",
        niter_pnp: int = 100,
        focal_length_estimation_method: str = "first_view_from_global_head",
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        if not load_pretrained_weights:
            raise ValueError("Fast3RWrapper currently requires load_pretrained_weights=True.")
        if gradient_checkpointing:
            print("[WARN] Fast3RWrapper ignores gradient_checkpointing; it is inference-only.")
        if torch_hub_force_reload:
            print("[WARN] torch_hub_force_reload is unused for Fast3R.")

        _ensure_fast3r_importable()
        from fast3r.utils.checkpoint_utils import load_model

        self.name = name
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.dtype = _resolve_inference_precision(dtype)
        self.niter_pnp = int(niter_pnp)
        self.focal_length_estimation_method = str(focal_length_estimation_method)
        self._pending_device = torch.device("cpu")

        print(f"Loading Fast3R from {pretrained_model_name_or_path} ...")
        self.model, self.lit_module = load_model(
            str(pretrained_model_name_or_path),
            device=torch.device("cpu"),
            is_lightning_checkpoint=False,
        )
        self.model.eval()
        self.lit_module.eval()

    def _apply(self, fn):
        super()._apply(fn)
        try:
            marker = torch.empty((), device=self._pending_device)
            self._pending_device = fn(marker).device
        except Exception:
            pass
        return self

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        try:
            self._pending_device = torch._C._nn._parse_to(*args, **kwargs)[0] or self._pending_device
        except Exception:
            pass
        return module

    @staticmethod
    def _views_to_fast3r_inputs(views: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        converted: List[Dict[str, object]] = []
        for idx, view in enumerate(views):
            data_norm_type = _first_data_norm_type(view)
            assert data_norm_type == "identity", (
                "Fast3R wrapper expects identity-normalized images in [0, 1]. "
                "Set configs/model/fast3r.yaml data_norm_type: identity."
            )

            img = view["img"]
            fast3r_img = img * 2.0 - 1.0
            true_shape = view.get("true_shape", None)
            if true_shape is None:
                true_shape = torch.tensor(
                    [[int(img.shape[-2]), int(img.shape[-1])]],
                    dtype=torch.int32,
                    device=img.device,
                )

            converted.append(
                {
                    "img": fast3r_img,
                    "true_shape": true_shape,
                    "idx": idx,
                    "instance": str(idx),
                }
            )
        return converted

    @staticmethod
    def _make_intrinsics(
        focal,
        batch_size: int,
        height: int,
        width: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        K = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        focal_tensor = torch.as_tensor(focal, dtype=dtype, device=device).reshape(-1)
        if focal_tensor.numel() == 1 and batch_size > 1:
            focal_tensor = focal_tensor.repeat(batch_size)
        if focal_tensor.numel() != batch_size:
            focal_tensor = torch.full((batch_size,), float(width), dtype=dtype, device=device)
        K[:, 0, 0] = focal_tensor
        K[:, 1, 1] = focal_tensor
        K[:, 0, 2] = float(width) * 0.5
        K[:, 1, 2] = float(height) * 0.5
        return K

    @staticmethod
    def _world_to_camera_points(points_world: torch.Tensor, T_c2w: torch.Tensor) -> torch.Tensor:
        R = T_c2w[:, :3, :3]
        t = T_c2w[:, :3, 3]
        return torch.einsum("bij,bhwj->bhwi", R.transpose(-1, -2), points_world - t[:, None, None, :])

    def forward(self, views):
        """Run Fast3R on all input views at once and return MapAnything-style outputs."""

        _ensure_fast3r_importable()
        from fast3r.dust3r.inference_multiview import inference
        from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

        if len(views) == 0:
            return []

        device = views[0]["img"].device
        self.model.to(device)
        self.model.eval()
        self.lit_module.eval()

        fast3r_views = self._views_to_fast3r_inputs(views)
        output = inference(
            fast3r_views,
            self.model,
            torch.device(device),
            dtype=self.dtype,
            verbose=False,
            profiling=False,
        )

        poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
            output["preds"],
            niter_PnP=self.niter_pnp,
            focal_length_estimation_method=self.focal_length_estimation_method,
        )

        num_views = len(output["preds"])
        batch_size = int(output["preds"][0]["pts3d_in_other_view"].shape[0])
        res = []
        for view_idx in range(num_views):
            pred = output["preds"][view_idx]
            points_world = pred["pts3d_in_other_view"].to(device=device, dtype=torch.float32)
            conf = pred["conf"].to(device=device, dtype=torch.float32)
            if conf.ndim == 4 and conf.shape[-1] == 1:
                conf = conf[..., 0]

            H, W = int(points_world.shape[1]), int(points_world.shape[2])
            poses = []
            focals = []
            for batch_idx in range(batch_size):
                pose = poses_c2w_batch[batch_idx][view_idx]
                poses.append(torch.as_tensor(pose, dtype=torch.float32, device=device))
                focal = None
                if batch_idx < len(estimated_focals) and view_idx < len(estimated_focals[batch_idx]):
                    focal = estimated_focals[batch_idx][view_idx]
                focals.append(float(focal) if focal is not None else float(W))

            T_c2w = torch.stack(poses, dim=0)
            pts3d_cam = self._world_to_camera_points(points_world, T_c2w)
            depth_along_ray = torch.linalg.norm(pts3d_cam, dim=-1, keepdim=True)
            ray_directions = pts3d_cam / depth_along_ray.clamp_min(1e-8)
            intrinsics = self._make_intrinsics(
                focals,
                batch_size,
                H,
                W,
                dtype=torch.float32,
                device=device,
            )

            res.append(
                {
                    "pts3d": points_world,
                    "pts3d_cam": pts3d_cam,
                    "ray_directions": ray_directions,
                    "intrinsics": intrinsics,
                    "depth_along_ray": depth_along_ray,
                    "cam_trans": T_c2w[:, :3, 3],
                    "cam_quats": mat_to_quat(T_c2w[:, :3, :3]),
                    "conf": conf,
                }
            )

        return res
