import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from copy import deepcopy
from huggingface_hub import PyTorchModelHubMixin
import numpy as np
from torch.utils.checkpoint import checkpoint

from mapanything.models.external.dinov2.hub.backbones import dinov2_vitl14_reg
from mapanything.models.external.pi3.layers.pos_embed import PositionGetter, RoPE2D
from mapanything.models.external.pi3.layers.block import BlockRope, PoseInjectBlock
from mapanything.models.external.pi3.layers.attention import FlashAttentionRope
from mapanything.models.external.dinov2.layers import Mlp, PatchEmbed
from mapanything.models.external.pi3.layers.camera_head import CameraHead
from mapanything.models.external.pi3.layers.conv_head import ConvHead
from mapanything.models.external.pi3.layers.transformer_head import TransformerDecoder, ContextOnlyTransformerDecoder

def se3_inverse(T):
    """
    Computes the inverse of a batch of SE(3) matrices.
    """

    if torch.is_tensor(T):
        R = T[..., :3, :3]
        t = T[..., :3, 3].unsqueeze(-1)
        R_inv = R.transpose(-2, -1)
        t_inv = -torch.matmul(R_inv, t)
        T_inv = torch.cat([
            torch.cat([R_inv, t_inv], dim=-1),
            torch.tensor([0, 0, 0, 1], device=T.device, dtype=T.dtype).repeat(*T.shape[:-2], 1, 1)
        ], dim=-2)
    else:
        R = T[..., :3, :3]
        t = T[..., :3, 3, np.newaxis]

        R_inv = np.swapaxes(R, -2, -1)
        t_inv = -R_inv @ t

        bottom_row = np.zeros((*T.shape[:-2], 1, 4), dtype=T.dtype)
        bottom_row[..., :, 3] = 1

        top_part = np.concatenate([R_inv, t_inv], axis=-1)
        T_inv = np.concatenate([top_part, bottom_row], axis=-2)
    return T_inv


def get_pixel(H, W):
    # get 2D pixels (u, v) for image_a in cam_a pixel space
    u_a, v_a = np.meshgrid(np.arange(W), np.arange(H))
    # u_a = np.flip(u_a, axis=1)
    # v_a = np.flip(v_a, axis=0)
    pixels_a = np.stack([
        u_a.flatten() + 0.5, 
        v_a.flatten() + 0.5, 
        np.ones_like(u_a.flatten())
    ], axis=0)
    return pixels_a


def homogenize_points(
    points,
):
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)



class Pi3X(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            ckpt=None,    
            use_multimodal=True,
            gradient_checkpointing=False,
            checkpoint_strategy="all",  # "all" or "global_only"
        ):
        super().__init__()

        self.use_multimodal = use_multimodal
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoint_strategy = checkpoint_strategy

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # Wrap encoder blocks with checkpointing
        if self.gradient_checkpointing:
            for i in range(len(self.encoder.blocks)):
                self.encoder.blocks[i] = self.wrap_module_with_gradient_checkpointing(
                    self.encoder.blocks[i]
                )

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        freq = 100
        self.rope = RoPE2D(freq=freq)
        self.position_getter = PositionGetter()

        # ----------------------
        #        Decoder
        # ----------------------
        dec_embed_dim = 1024
        dec_num_heads = 16
        mlp_ratio = 4
        dec_depth = 36      
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
        ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # -----------------------
        #       multi-modal
        # -----------------------
        if use_multimodal:
            ## Depth encoder
            self.depth_encoder = deepcopy(self.encoder)
            del self.depth_encoder.patch_embed
            self.depth_encoder.patch_embed = PatchEmbed(img_size=224, patch_size=14, in_chans=2, embed_dim=1024)
            self.depth_emb = nn.Parameter(torch.zeros(1, 1, 1024))

            ## Ray embedding
            self.ray_embed = PatchEmbed(img_size=224, patch_size=14, in_chans=2, embed_dim=1024)
            nn.init.constant_(self.ray_embed.proj.weight, 0)
            nn.init.constant_(self.ray_embed.proj.bias, 0)

            ## Pose inject blocks
            self.pose_inject_blk = nn.ModuleList([PoseInjectBlock(
                dim=1024,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
            ) for _ in range(5)])


        # ------------------------------
        #           Head
        # ------------------------------
        ## --------------- Point ---------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=1024,
            rope=self.rope,
            use_checkpoint=self.gradient_checkpointing
            # use_checkpoint=False,
        )
        # self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)
        self.point_head = ConvHead(
                num_features=4, 
                dim_in=dec_embed_dim,
                # projects=nn.Linear(1024, 1024),
                projects=nn.Identity(),
                dim_out=[2, 1], 
                dim_proj=1024,
                dim_upsample=[256, 128, 64],
                dim_times_res_block_hidden=2,
                num_res_blocks=2,
                res_block_norm='group_norm',
                last_res_blocks=0,
                last_conv_channels=32,
                last_conv_size=1,
                using_uv=True
            )

        ## --------------- Camera ---------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            # use_checkpoint=self.gradient_checkpointing
            use_checkpoint=False,
        )
        self.camera_head = CameraHead(dim=512)

        ## --------------- Metric ---------------
        self.metric_token = nn.Parameter(torch.randn(1, 1, 2*self.dec_embed_dim))
        self.metric_decoder = ContextOnlyTransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=512,
            dec_num_heads=8,                # 8
            out_dim=512,
            rope=self.rope,
            # use_checkpoint=self.gradient_checkpointing
            use_checkpoint=False
        )
        self.metric_head = nn.Linear(512, 1)
        nn.init.normal_(self.metric_token, std=1e-6)


        ## -------------- Conf ------------------
        self.conf_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=1024,
            rope=self.rope,
            use_checkpoint=self.gradient_checkpointing
            # use_checkpoint=False
        )
        self.conf_head = ConvHead(
            num_features=4, 
            dim_in=dec_embed_dim,
            # projects=nn.Linear(1024, 1024),
            projects=nn.Identity(),
            dim_out=[1], 
            dim_proj=1024,
            dim_upsample=[256, 128, 64],
            dim_times_res_block_hidden=2,
            num_res_blocks=2,
            res_block_norm='group_norm',
            last_res_blocks=0,
            last_conv_channels=32,
            last_conv_size=1,
            using_uv=True
        )

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)


    def wrap_module_with_gradient_checkpointing(self, module: nn.Module):
        class _CheckpointingWrapper(module.__class__):
            _restore_cls = module.__class__
            def forward(self, *args, **kwargs):
                return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)
        module.__class__ = _CheckpointingWrapper
        return module
    

    def disable_multimodal(self, free_cuda_cache: bool = True):
        """
        Disables multimodal branches and releases their modules/parameters.
        Use this when no multimodal conditions are provided.
        """
        self.use_multimodal = False
        for attr in ("depth_encoder", "depth_emb", "ray_embed", "pose_inject_blk"):
            if hasattr(self, attr):
                delattr(self, attr)

        if free_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()


    def forward(
        self,
        imgs,
        depths=None,
        intrinsics=None,
        rays=None,
        poses=None,
        pose_mask=None,
        with_prior=None,
        overall_prob=1.0,
        ray_dirs_prob=0.0,
        depth_prob=0.0,
        cam_prob=0.0,
    ):
        """
        Forward pass with optional multimodal conditions.

        Args:
            imgs (torch.Tensor): Input RGB images valued in [0, 1].
                Shape: (B, N, 3, H, W).
            intrinsics (torch.Tensor, optional): Camera intrinsic matrices.
                Shape: (B, N, 3, 3).
                Values are in pixel coordinates (not normalized).
            rays (torch.Tensor, optional): Pre-computed ray directions (unit vectors).
                Shape: (B, N, H, W, 3).
                Can replace `intrinsics` as a geometric condition.
            poses (torch.Tensor, optional): Camera-to-World matrices.
                Shape: (B, N, 4, 4).
                Coordinate system: OpenCV convention (Right-Down-Forward).
            depths (torch.Tensor, optional): Ground truth or prior depth maps.
                Shape: (B, N, H, W).
                Invalid values (e.g., sky or missing data) should be set to 0.
            mask_add_depth (torch.Tensor, optional): Mask for depth condition.
                Shape: (B, N, N).
            mask_add_ray (torch.Tensor, optional): Mask for ray/intrinsic condition.
                Shape: (B, N, N).
            mask_add_pose (torch.Tensor, optional): Mask for pose condition.
                Shape: (B, N, N).
                Note: Requires at least two frames to be True to establish a meaningful
                coordinate system (absolute pose for a single frame provides no relative constraint).

        Returns:
            dict: Model outputs containing 'points', 'conf', etc.
        """
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        # encode
        hidden, poses_, use_depth_mask, use_pose_mask, norm_factor = self.encode(
            imgs,
            with_prior=with_prior,
            depths=depths,
            intrinsics=intrinsics,
            poses=poses,
            pose_mask=pose_mask,
            rays=rays,
            overall_prob=overall_prob,
            ray_dirs_prob=ray_dirs_prob,
            depth_prob=depth_prob,
            cam_prob=cam_prob,
        )
        hidden = hidden.reshape(B, N, -1, self.dec_embed_dim)

        # decode
        hidden, pos = self.decode(hidden, N, H, W, poses_, use_pose_mask)

        # # head
        outputs = self.forward_head(hidden, pos, B, N, H, W, patch_h, patch_w)

        return outputs
    
    def encode(
        self,
        imgs,
        with_prior=None,
        depths=None,
        rays=None,
        intrinsics=None,
        poses=None,
        pose_mask=None,
        overall_prob=1.0,
        ray_dirs_prob=0.0,
        depth_prob=0.0,
        cam_prob=0.0,
    ):
        B, N, _, H, W = imgs.shape
        device = imgs.device

        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)["x_norm_patchtokens"]

        if self.use_multimodal:
            with torch.amp.autocast(device_type='cuda', enabled=False):

                if with_prior is False:
                    p_depth = 0.0
                    p_ray = 0.0
                    p_pose = 0.0
                else:
                    # with_prior=None: training mode, sample according to task probabilities.
                    # with_prior=True: explicit prior mode, also use provided task probabilities.
                    # This is different from the original public inference code, but matches
                    # the training need where each task has its own prior probabilities.
                    if torch.rand(1, device=device) < float(overall_prob):
                        p_depth = float(depth_prob)
                        p_ray = float(ray_dirs_prob)
                        p_pose = float(cam_prob)
                    else:
                        p_depth = 0.0
                        p_ray = 0.0
                        p_pose = 0.0

                if depths is None:
                    p_depth = 0.0
                    depths = torch.zeros((B, N, H, W), device=imgs.device)

                if rays is not None:
                    rays = rays[..., :2] / (rays[..., 2:3] + 1e-6)
                else:
                    if intrinsics is None:
                        p_ray = 0.0
                        rays = torch.zeros((B, N, H, W, 2), device=imgs.device)
                    else:
                        pix = torch.from_numpy(get_pixel(H, W).T.reshape(H, W, 3)).to(device).float()[None].repeat(B, 1, 1, 1)
                        rays = torch.einsum('bnij, bhwj -> bnhwi', torch.inverse(intrinsics), pix)[..., :2]
                        # rays = F.normalize(rays, dim=-1).reshape(B, N, H, W, 3)                   # don't normalize, so the pred['xy'] is the same as input rays

                if poses is None:
                    p_pose = 0.0
                    poses = torch.eye(4, device=device)[None, None].repeat(B, N, 1, 1)
                else:
                    assert rays is not None                     # rays should be along with poses
                    
                mask_add_depth = torch.rand((B, N), device=device) <= p_depth
                mask_add_ray = torch.rand((B, N), device=device) <= p_ray
                if pose_mask is not None:
                    if pose_mask.shape != (B, N):
                        raise ValueError("pose_mask must have shape [B, N]")
                    mask_add_pose = pose_mask.to(device=device, dtype=torch.bool)
                else:
                    mask_add_pose = torch.rand((B, N), device=device) <= p_pose

                # pose is injected relatively. so at least two frame should be true.
                num_valid_pose = mask_add_pose.sum(dim=1)
                bad_indices = (num_valid_pose == 1)
                mask_add_pose[bad_indices] = False

                # normalize depth and pose
                normalized_depths, dep_median = self.normalize_depth(depths, method='mean')
                scale_aug = 0.8 + torch.rand((B,), device=device) * 0.4
                normalized_depths /= scale_aug.view(B, 1, 1, 1)
                dep_median *= scale_aug

                depths_masks = (normalized_depths > 0).float()
                depths_masks = depths_masks.reshape(B*N, 1, H, W)

                if pose_mask is not None:
                    # Choose the first actually available pose as reference;
                    # masked/missing poses must not define the coordinate frame.
                    first_valid = mask_add_pose.float().argmax(dim=1)
                    ref_poses = poses[torch.arange(B, device=device), first_valid]
                else:
                    ref_poses = poses[:, 0]
                poses_ = torch.einsum('bij, bnjk -> bnik', se3_inverse(ref_poses), poses)
                poses_[..., :3, 3] /= dep_median.view(B, 1, 1)

                # noramlize for the batch not using depth
                use_depth_batch_mask = mask_add_depth.sum(dim=1) > 0
                if (~use_depth_batch_mask).sum() > 0 and N > 1:
                    if pose_mask is not None:
                        pose_dist = poses_[..., :3, 3].norm(dim=-1)
                        pose_valid = mask_add_pose.to(pose_dist.dtype)
                        pose_count = pose_valid.sum(dim=1)
                        pose_scale = (pose_dist * pose_valid).sum(dim=1) / pose_count.clamp_min(1.0)
                        static_threshold = 2e-2
                        masked_max = (pose_dist * pose_valid).max(dim=1)[0]
                        is_static_mask = masked_max < static_threshold
                        has_pose_baseline = pose_count >= 2
                    else:
                        pose_scale_per_view = poses_[..., 1:, :3, 3].norm(dim=-1)
                        static_threshold = 2e-2
                        is_static_mask = pose_scale_per_view.max(dim=1)[0] < static_threshold
                        pose_scale = pose_scale_per_view.mean(dim=1)
                        has_pose_baseline = torch.ones(B, dtype=torch.bool, device=device)
                    scale_aug = 0.8 + torch.rand((B,), device=device) * 0.4
                    pose_scale *= scale_aug

                    final_moving_mask = (
                        (~use_depth_batch_mask) & (~is_static_mask) & has_pose_baseline
                    )
                    poses_[final_moving_mask, ..., :3, 3] /= (pose_scale.view(B, 1, 1)[final_moving_mask] + 1e-8)
                    normalized_depths[final_moving_mask] /= (pose_scale.view(B, 1, 1, 1)[final_moving_mask] + 1e-8)

                    dep_median[final_moving_mask] *= pose_scale[final_moving_mask]
                
                # if with_prior is None:
                #     add_noise_batch = torch.rand(B) > 0.5
                # else:
                #     add_noise_batch = torch.rand(B) > 1   

                # if N > 1:
                #     poses_[add_noise_batch, 1:] = add_randomized_smooth_pose_noise_torch(poses_[add_noise_batch, 1:])

                normalized_depths = normalized_depths.reshape(B*N, 1, H, W)

            if mask_add_depth.sum() > 0:
                depth_emb = self.depth_encoder(
                    torch.cat([normalized_depths, depths_masks], dim=1),
                    is_training=True,
                )["x_norm_patchtokens"] + self.depth_emb
            else:
                depth_emb = torch.zeros_like(hidden)

            if mask_add_ray.sum() > 0:
                ray_emb = self.ray_embed(
                    rays.reshape(B * N, H, W, 2).permute(0, 3, 1, 2)
                )
            else:
                ray_emb = torch.zeros_like(hidden)
            
            
            use_depth_mask = mask_add_depth
            use_pose_mask = mask_add_pose

            # hidden = hidden + ray_emb * mask_add_ray.reshape(B*N, 1, 1)
            # hidden = hidden + depth_emb * mask_add_depth.reshape(B*N, 1, 1)

            depth_mask = mask_add_depth.reshape(B * N, 1, 1)
            ray_mask = mask_add_ray.reshape(B * N, 1, 1)
            
            depth_emb = torch.where(depth_mask, depth_emb, torch.zeros_like(depth_emb))
            ray_emb = torch.where(ray_mask, ray_emb, torch.zeros_like(ray_emb))

            hidden = hidden + ray_emb
            hidden = hidden + depth_emb

            return hidden, poses_, use_depth_mask, use_pose_mask, dep_median
        
        return hidden, None, None, None, None
    
    def _chunked_conv_head(self, head, feat, patch_h, patch_w, chunk_size=64):
        BN = feat.shape[0]
        if BN <= chunk_size:
            return head(feat, patch_h=patch_h, patch_w=patch_w)
        outputs = [[] for _ in range(len(head.output_block))] if isinstance(head.output_block, nn.ModuleList) else []
        for i in range(0, BN, chunk_size):
            chunk_out = head(feat[i:i+chunk_size], patch_h=patch_h, patch_w=patch_w)
            if isinstance(chunk_out, list):
                for j, o in enumerate(chunk_out):
                    outputs[j].append(o)
            else:
                outputs.append(chunk_out)
        if isinstance(outputs[0], list):
            return [torch.cat(parts, dim=0) for parts in outputs]
        return torch.cat(outputs, dim=0)

    def forward_head(self, hidden, pos, B, N, H, W, patch_h, patch_w):
        device = hidden.device
        hw = patch_h*patch_w+self.patch_start_idx

        # decode point
        ret_point = self.point_decoder(hidden, xpos=pos)

        # decode camera
        ret_camera = self.camera_decoder(hidden, xpos=pos)

        # decode metric
        pos_hw = pos.reshape(B, N*hw, -1)
        ret_metric = self.metric_decoder(self.metric_token.repeat(B, 1, 1), hidden.reshape(B, N*hw, -1), xpos=pos_hw[:, 0:1], ypos=pos_hw)

        # decode conf
        ret_conf = self.conf_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            point_feat = ret_point[:, self.patch_start_idx:].float()
            xy, z = self._chunked_conv_head(self.point_head, point_feat, patch_h, patch_w)
            del point_feat

            # xy = xy.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)
            # z = z.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)

            # # z = torch.exp(z.clamp(max=15.0))
            # z = torch.nan_to_num(z.float(), nan=0.0, posinf=15.0, neginf=-15.0)
            # z = torch.exp(z.clamp(min=-15.0, max=15.0))

            # local_points = torch.cat([xy * z, z], dim=-1)
            # rays = F.normalize(torch.cat([xy, torch.ones_like(z)], dim=-1), dim=-1)


            xy = xy.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)
            z = z.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)

            xy = torch.nan_to_num(xy.float(), nan=0.0, posinf=1e4, neginf=-1e4,)
            xy = xy.clamp(min=-1e4, max=1e4)

            z = torch.nan_to_num(z.float(), nan=0.0, posinf=15.0, neginf=-15.0,)
            z = torch.exp(z.clamp(min=-15.0, max=15.0))

            local_points = torch.cat([xy * z, z], dim=-1)
            local_points = torch.nan_to_num(local_points,nan=0.0, posinf=1e6, neginf=-1e6,)

            ray_input = torch.cat([xy, torch.ones_like(z)], dim=-1)
            ray_input = torch.nan_to_num(ray_input, nan=0.0, posinf=1e4, neginf=-1e4)
            rays = F.normalize(ray_input, dim=-1, eps=1e-6)
            rays = torch.nan_to_num(rays, nan=0.0, posinf=1.0, neginf=-1.0)

            camera_poses = self.camera_head(ret_camera[:, self.patch_start_idx:].float(), patch_h, patch_w).reshape(B, N, 4, 4)

            # metric = self.metric_head(ret_metric.float()).reshape(B).exp()
            metric_log = self.metric_head(ret_metric.float()).reshape(B)
            metric_log = torch.nan_to_num(metric_log, nan=0.0, posinf=10.0, neginf=-10.0)
            metric = metric_log.clamp(min=-10.0, max=10.0).exp()

            # conf
            conf_feat = ret_conf[:, self.patch_start_idx:].float()
            conf = self._chunked_conv_head(self.conf_head, conf_feat, patch_h, patch_w)[0]
            del conf_feat
            conf = conf.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)

            # # points
            # points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3] * metric.view(B, 1, 1, 1, 1)

            # # convert camera poses to metric
            # camera_poses[..., :3, 3] = camera_poses[..., :3, 3] * metric.view(B, 1, 1)

            # # convert local_points to metric
            # local_points = local_points * metric.view(B, 1, 1, 1, 1)

            # convert local_points to metric
            metric_points = metric.view(B, 1, 1, 1, 1)
            metric_pose_t = metric.view(B, 1, 1, 1)

            local_points_metric = local_points * metric_points
            local_points_metric = torch.nan_to_num(local_points_metric, nan=0.0, posinf=1e6, neginf=-1e6,)

            # build metric camera poses without inplace modification
            camera_R = camera_poses[..., :3, :3]
            camera_t = camera_poses[..., :3, 3:4] * metric_pose_t
            camera_bottom = camera_poses[..., 3:4, :]

            camera_poses_metric = torch.cat(
                [
                    torch.cat([camera_R, camera_t], dim=-1),
                    camera_bottom,
                ],
                dim=-2,
            )
            camera_poses_metric = torch.nan_to_num(camera_poses_metric,nan=0.0, posinf=1e6, neginf=-1e6,)

            # points in metric world frame
            points = torch.einsum(
                'bnij, bnhwj -> bnhwi',
                camera_poses_metric,
                homogenize_points(local_points_metric),
            )[..., :3]

            points = torch.nan_to_num(points, nan=0.0, posinf=1e6, neginf=-1e6,)

            camera_poses = camera_poses_metric
            local_points = local_points_metric

        return dict(
            points=points,
            local_points=local_points,
            rays=rays,
            conf=conf,
            camera_poses=camera_poses,  
            metric=metric,
        )


    def decode(self, hidden, N, H, W, poses, use_pose_mask):
        device = hidden.device

        if len(hidden.shape) == 4:
            B, N, hw, _ = hidden.shape
        else:
            BN, hw, _ = hidden.shape
            B = BN // N

        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]
        pose_inject_blk_idx = 0

        pos = self.position_getter(B*N, H//self.patch_size, W//self.patch_size, hidden.device)
        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos_patch = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos_patch], dim=1)

        if self.use_multimodal:
            if use_pose_mask.sum() == B * N:
                pose_inject_mask = None
            else:
                view_interaction_mask = use_pose_mask.unsqueeze(2) & use_pose_mask.unsqueeze(1)
                token_interaction_mask = view_interaction_mask.repeat_interleave(hw - self.patch_start_idx, dim=1)
                token_interaction_mask = token_interaction_mask.repeat_interleave(hw - self.patch_start_idx, dim=2)
                pose_inject_mask = token_interaction_mask[:, None]

        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            # hidden = blk(hidden, xpos=pos)

            do_checkpoint = False
            if self.gradient_checkpointing:
                if self.checkpoint_strategy == 'all':
                    do_checkpoint = True
                elif self.checkpoint_strategy == 'global_only':
                    if i % 2 != 0:
                        do_checkpoint = True

            # if self.training and do_checkpoint:
            #     hidden = checkpoint(blk, hidden, xpos=pos, attn_mask=None, use_reentrant=False)
            # else:
            #     hidden = blk(hidden, xpos=pos)

            pos = pos.to(device=hidden.device, dtype=torch.long).contiguous().detach().clone()
            if self.training and do_checkpoint:
                def run_blk(x, xpos, blk=blk):
                    return blk(x, xpos=xpos)
                hidden = checkpoint(run_blk, hidden, pos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=pos)

            if self.use_multimodal:
                if i in [1, 9, 17, 25, 33] and use_pose_mask.sum() > 0:
                    hidden = hidden.reshape(B, N, -1, 1024)
                    poses_feat = self.pose_inject_blk[pose_inject_blk_idx](hidden[..., self.patch_start_idx:, :].reshape(B, N*(hw-self.patch_start_idx), -1), poses, H, W, H//14, W//14, attn_mask=pose_inject_mask).reshape(B, N, -1, 1024)
                    # hidden[..., self.patch_start_idx:, :] += poses_feat * use_pose_mask.view(B, N, 1, 1)

                    patch_hidden = hidden[..., self.patch_start_idx:, :]
                    patch_hidden = patch_hidden + poses_feat * use_pose_mask.view(B, N, 1, 1)
                    hidden = torch.cat([hidden[..., :self.patch_start_idx, :], patch_hidden], dim=2)

                    hidden = hidden.reshape(B, N*hw, -1)
                    pose_inject_blk_idx += 1

            if i == len(self.decoder) - 2:
                temp_features = hidden.clone().reshape(B*N, hw, -1)

        concatenated = torch.cat((temp_features, hidden.reshape(B*N, hw, -1)), dim=-1)

        return concatenated, pos.reshape(B*N, hw, -1)
    
    
    def normalize_depth(self, depths: torch.Tensor, method: str = 'median') -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalizes a batch of depth maps using either median or mean normalization.

        Args:
            depths (torch.Tensor): A batch of depth maps with shape [B, N, H, W].
                                Non-positive values are treated as invalid depth data.
            method (str, optional): The normalization method to use.
                                    Can be 'median' or 'mean'. Defaults to 'median'.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The normalized depth maps.
                - The normalization factors (medians or means) used for each batch element.

        Raises:
            ValueError: If the method is not 'median' or 'mean'.
        """
        # 确保输入是 torch.Tensor
        if not isinstance(depths, torch.Tensor):
            depths = torch.tensor(depths, dtype=torch.float32)

        if method not in ['median', 'mean']:
            raise ValueError(f"Invalid normalization method: '{method}'. Choose 'median' or 'mean'.")

        B, N, H, W = depths.shape
        epsilon = 1e-8

        # Create a mask for valid depth values (positive values)
        valid_depths = torch.where(depths > 0, depths, float('nan'))
        valid_depths_reshaped = valid_depths.view(B, -1)

        if method == 'median':
            # Calculate the median for each depth map in the batch
            factors, _ = torch.nanmedian(valid_depths_reshaped, dim=1)
        elif method == 'mean':
            # Calculate the mean for each depth map in the batch
            factors = torch.nanmean(valid_depths_reshaped, dim=1)
        
        # Handle cases where all values might be NaN (e.g., all depths are 0 or negative)
        # In such cases, use 1.0 as the normalization factor to prevent division by zero.
        factors = torch.nan_to_num(factors, nan=1.0)
        
        # Reshape factors for broadcasting during division
        factors_for_division = factors.view(B, 1, 1, 1)

        # Perform normalization, adding a small epsilon to prevent division by zero
        normalized_depths = depths / (factors_for_division + epsilon)

        return normalized_depths, factors.reshape(-1)
