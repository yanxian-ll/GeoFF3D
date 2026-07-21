import torch
import torch.nn as nn
import torch.nn.functional as F


class CameraHead(nn.Module):
    """
    Camera decoder adapted for MapAnything.

    Input:
        feat: (B, N, C) or (B*N, C)

    Output:
        dict with:
            - pose_encoding: (B, N, 9) = [t(3), quat_xyzw(4), fov_h, fov_w]
            - translation:   (B, N, 3)
            - quaternion:    (B, N, 4)
            - fov_hw:        (B, N, 2)
    """

    def __init__(
        self,
        dim_in: int,
        hidden_dim: int = None,
        num_layers: int = 2,
        predict_translation: bool = False,
        predict_rotation: bool = False,
        predict_fov: bool = True,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = dim_in

        layers = []
        in_dim = dim_in
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.predict_translation = predict_translation
        self.predict_rotation = predict_rotation
        self.predict_fov = predict_fov

        self.fc_t = nn.Linear(hidden_dim, 3) if predict_translation else None
        self.fc_qvec = nn.Linear(hidden_dim, 4) if predict_rotation else None
        self.fc_fov = nn.Linear(hidden_dim, 2) if predict_fov else None

    def forward(self, feat, camera_encoding=None):
        """
        Args:
            feat: (B, N, C) or (B*N, C)
            camera_encoding: optional tensor with shape (B, N, 9)
                If provided, any disabled branch can reuse values from it.

        Returns:
            dict with pose_encoding / translation / quaternion / fov_hw
        """
        reshape_back = False
        if feat.ndim == 3:
            B, N, C = feat.shape
            feat_flat = feat.reshape(B * N, C)
            reshape_back = True
        elif feat.ndim == 2:
            feat_flat = feat
            B = N = None
        else:
            raise ValueError(f"Invalid feat shape: {feat.shape}")

        x = self.backbone(feat_flat)

        # translation
        if self.fc_t is not None:
            out_t = self.fc_t(x.float())
            if reshape_back:
                out_t = out_t.reshape(B, N, 3)
        else:
            if camera_encoding is not None:
                out_t = camera_encoding[..., :3]
            else:
                out_t = None

        # quaternion (xyzw)
        if self.fc_qvec is not None:
            out_q = self.fc_qvec(x.float())
            out_q = F.normalize(out_q, dim=-1)
            if reshape_back:
                out_q = out_q.reshape(B, N, 4)
        else:
            if camera_encoding is not None:
                out_q = camera_encoding[..., 3:7]
            else:
                out_q = None

        # fov_h, fov_w
        if self.fc_fov is not None:
            out_fov = self.fc_fov(x.float())
            out_fov = F.relu(out_fov)
            if reshape_back:
                out_fov = out_fov.reshape(B, N, 2)
        else:
            if camera_encoding is not None:
                out_fov = camera_encoding[..., 7:9]
            else:
                out_fov = None

        out = {}
        if out_t is not None:
            out["translation"] = out_t
        if out_q is not None:
            out["quaternion"] = out_q
        if out_fov is not None:
            out["fov_hw"] = out_fov
        return out
    