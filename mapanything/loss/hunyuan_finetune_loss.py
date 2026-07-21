import torch
import torch.nn as nn

from hunyuanworld_mirror.losses.container import LossContainer
from hunyuanworld_mirror.losses.camera import CameraLoss
from hunyuanworld_mirror.losses.point import PointLoss
from hunyuanworld_mirror.losses.depth import DepthLoss
from hunyuanworld_mirror.losses.normal import NormalLoss


class HunyuanFineTuneLoss(nn.Module):
    def __init__(
        self,
        enable_cam=True,
        enable_pts=True,
        enable_depth=True,
        enable_norm=True,
        enable_gs=False,
    ):
        super().__init__()
        if enable_gs:
            raise NotImplementedError(
                "GS losses are not yet adapted to the MapAnything Hunyuan wrapper. "
                "Use enable_gs=False."
            )

        self.loss_container = LossContainer(
            enable_cam=enable_cam,
            enable_pts=enable_pts,
            enable_depth=enable_depth,
            enable_norm=enable_norm,
            enable_gs=False,
            losses=[
                CameraLoss(),
                PointLoss(gamma=1.0, alpha=0.2, gradient_loss_fn="normal", valid_range=0.98),
                DepthLoss(gamma=1.0, alpha=0.2, gradient_loss_fn="grad", valid_range=0.98),
                NormalLoss(loss_type="AL", real_weight=1.0, pseudo_weight=1.0, ignore_datasets=[]),
            ],
            weights=[5.0, 1.0, 1.0, 1.0],
        )

    def _stack_batch(self, batch):
        device = batch[0]["img"].device
        gts = {
            "img": torch.stack([v["img"] for v in batch], dim=1),
            "camera_poses": torch.stack([v["camera_pose"].to(device) for v in batch], dim=1),
            "camera_intrs": torch.stack([v["camera_intrinsics"].to(device) for v in batch], dim=1),
            "depthmap": torch.stack([v["depthmap"].to(device) for v in batch], dim=1),
            "pts3d": torch.stack([v["pts3d"].to(device) for v in batch], dim=1),
            "valid_mask": torch.stack([v["valid_mask"].to(device) for v in batch], dim=1),
        }
        if all("normals" in v for v in batch):
            gts["normals"] = torch.stack([v["normals"].to(device) for v in batch], dim=1)

        dataset_field = batch[0].get("dataset", "unknown")
        if isinstance(dataset_field, list):
            gts["dataset"] = dataset_field
        else:
            gts["dataset"] = [v.get("dataset", "unknown") for v in batch]
        return gts

    def _stack_preds(self, preds):
        pred_dict = {
            "camera_params": torch.stack([p["pred_camera_params"] for p in preds], dim=1),
            "camera_poses": torch.stack([p["pred_c2w"] for p in preds], dim=1),
            "camera_intrs": torch.stack([p["pred_intrinsics"] for p in preds], dim=1),
            "depth": torch.stack([p["depth_z"] for p in preds], dim=1),
            "depth_conf": torch.stack([p["conf"] for p in preds], dim=1),
            "pts3d": torch.stack([p["pts3d"] for p in preds], dim=1),
            "pts3d_conf": torch.stack([p["pts3d_conf"] for p in preds], dim=1),
        }
        if all("normals" in p for p in preds):
            pred_dict["normals"] = torch.stack([p["normals"] for p in preds], dim=1)
            pred_dict["normals_conf"] = torch.stack([p["normals_conf"] for p in preds], dim=1)
        return pred_dict

    def forward(self, batch, preds, **kwargs):
        gts = self._stack_batch(batch)
        pred_dict = self._stack_preds(preds)
        loss, details = self.loss_container(gts, pred_dict)

        clean_details = {}
        for k, v in details.items():
            if torch.is_tensor(v):
                clean_details[k] = float(v.detach())
            else:
                clean_details[k] = v
        return loss, clean_details
