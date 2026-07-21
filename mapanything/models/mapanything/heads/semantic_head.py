# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Semantic Segmentation Head for MapAnything.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from uniception.models.prediction_heads.dpt import DPTRegressionProcessor, DPTFeature


class SemanticSegmentationHead(nn.Module):
    """
    Semantic segmentation head that takes multi-scale features from DPT
    and produces per-pixel semantic class predictions.

    Args:
        num_classes (int): Number of semantic classes
        feature_dim (int): Dimension of input features
        feature_scales (List[int]): List of feature scales to use
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 256,
        feature_scales: Optional[List[int]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.target_feature_dim = feature_dim  # Expected feature dim for internal layers

        if feature_scales is None:
            feature_scales = [32, 16, 8, 4]

        self.feature_scales = feature_scales

        # Input projections - will be built lazily in first forward pass
        self._input_proj = None

        # Build projection layers for each feature scale (using target dim)
        self.scale_convs = nn.ModuleList()
        for scale in feature_scales:
            # Use DPT-style upsampling
            self.scale_convs.append(
                nn.Sequential(
                    nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature_dim),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False),
                )
            )

        # Fusion convolutions
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim * len(feature_scales), 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Final classification layer
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def _get_input_proj(self, in_channels: int, dtype: torch.dtype, device: torch.device) -> nn.Module:
        """Get or create input projection layer for given channel dimension."""
        if self._input_proj is None or self._input_proj[0].in_channels != in_channels:
            self._input_proj = nn.Sequential(
                nn.Conv2d(in_channels, self.target_feature_dim, kernel_size=1),
                nn.BatchNorm2d(self.target_feature_dim),
                nn.ReLU(inplace=True),
            ).to(dtype=dtype, device=device)
            # Register as submodule so parameters are learned and included in state_dict
            self.add_module("_input_proj", self._input_proj)
        # Ensure the projection matches input dtype/device (handles bfloat16 etc.)
        elif self._input_proj[0].weight.dtype != dtype:
            self._input_proj = self._input_proj.to(dtype=dtype, device=device)
        return self._input_proj

    def forward(self, dense_head_inputs: List[torch.Tensor], img_shape: tuple) -> torch.Tensor:
        """
        Forward pass for semantic segmentation.

        Args:
            dense_head_inputs: List of multi-scale features from DPT
            img_shape: Target image shape (height, width)

        Returns:
            Semantic segmentation logits of shape (B, num_classes, H, W)
        """
        target_h, target_w = img_shape

        # Project and resize each scale
        scaled_features = []
        for i, conv in enumerate(self.scale_convs):
            if i < len(dense_head_inputs):
                feat = dense_head_inputs[i]

                # Handle different input formats:
                # - (B, C, H*W): flattened spatial dimensions (common for DPT)
                # - (B, C, H, W): already has spatial dimensions
                if feat.dim() == 3:
                    # Reshape from (B, C, H*W) to (B, C, H, W)
                    B, C, HW = feat.shape
                    H = W = int(HW ** 0.5)
                    if H * W != HW:
                        # Handle non-square case
                        H = int((HW / (target_w / target_h)) ** 0.5)
                        W = HW // H
                    feat = feat.view(B, C, H, W)
                elif feat.dim() == 4:
                    # Already in (B, C, H, W) format
                    pass
                else:
                    raise ValueError(f"Expected 3D or 4D tensor, got {feat.dim()}D")

                # Project input channels to target feature dimension
                in_channels = feat.shape[1]
                if in_channels != self.target_feature_dim:
                    proj = self._get_input_proj(in_channels, feat.dtype, feat.device)
                    feat = proj(feat)

                # Project and upsample to target size
                feat = conv(feat)
                feat = nn.functional.interpolate(
                    feat, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
                scaled_features.append(feat)

        # Concatenate all scales
        if scaled_features:
            fused = torch.cat(scaled_features, dim=1)
        else:
            raise ValueError("No features provided to semantic segmentation head")

        # Apply fusion convolutions
        fused = self.fusion_conv(fused)

        # Final classification
        logits = self.classifier(fused)

        return logits


class MultiTaskHead(nn.Module):
    """
    Multi-task head that outputs both geometric predictions (depth/pointmap)
    and semantic segmentation predictions.

    Args:
        num_classes (int): Number of semantic classes
        geometric_head: Existing geometric prediction head (e.g., DPT with adaptor)
        semantic_head: Semantic segmentation head
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.semantic_head = SemanticSegmentationHead(
            num_classes=num_classes,
            feature_dim=feature_dim,
        )

    def forward(
        self,
        dense_head_inputs: List[torch.Tensor],
        scale_head_inputs: torch.Tensor,
        img_shape: tuple,
        memory_efficient_inference: bool = False,
        minibatch_size: Optional[int] = None,
    ):
        """
        Forward pass producing both geometric and semantic outputs.

        Returns:
            Tuple of (geometric_outputs, semantic_logits)
            - geometric_outputs: Contains dense_final_outputs, pose_final_outputs, scale_final_output
            - semantic_logits: (B, num_classes, H, W) semantic predictions
        """
        raise NotImplementedError("MultiTaskHead should be used with specific downstream head implementation")
