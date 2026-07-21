#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extended VGGT test script with:
1. Global-layer attention visualization.
2. Cross-view patch correspondence visualization.
3. LightGlue-based sparse matching + multi-view track construction.
4. Tie-point prior injection into VGGT middle global attention layers.
5. Bias visualization.
6. Delta-logits / delta-attention comparison between injected and baseline runs.
7. Regular-grid token correspondence visualization from global attention.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import types
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from mapanything.models import init_model_from_config
from mapanything.utils.image import load_images

from lightglue import ALIKED, DISK, LightGlue, SIFT, SuperPoint
from lightglue.utils import rbd


# -----------------------------------------------------------------------------
# Torch hub offline redirect
# -----------------------------------------------------------------------------
def setup_offline_torch_hub(
    torch_hub_dir: Optional[str],
    local_dino_repo: Optional[str],
) -> None:
    if torch_hub_dir:
        torch.hub.set_dir(torch_hub_dir)

    if not local_dino_repo:
        return

    original_torch_hub_load = torch.hub.load

    def offline_torch_hub_load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == "facebookresearch/dinov2":
            print(f"[Info] Redirecting DINOv2 torch.hub.load -> {local_dino_repo}")
            repo_or_dir = local_dino_repo
            kwargs["source"] = "local"
        return original_torch_hub_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = offline_torch_hub_load


# -----------------------------------------------------------------------------
# Simple data structures
# -----------------------------------------------------------------------------
@dataclass
class PairMatch:
    view_a: int
    view_b: int
    kp_idx_a: int
    kp_idx_b: int
    xy_a: Tuple[float, float]
    xy_b: Tuple[float, float]
    confidence: float
    distance: float
    ratio_score: float


@dataclass
class TrackObservation:
    view_idx: int
    kp_idx: int
    xy: Tuple[float, float]
    response: float


@dataclass
class Track:
    track_id: int
    observations: List[TrackObservation]
    confidence: float


@dataclass
class SimpleKeyPoint:
    pt: Tuple[float, float]
    response: float


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(folder: str) -> List[str]:
    paths = []
    for name in sorted(os.listdir(folder)):
        if Path(name).suffix.lower() in SUPPORTED_EXTS:
            paths.append(os.path.join(folder, name))
    if not paths:
        raise FileNotFoundError(f"No supported images found in: {folder}")
    return paths


def tensor_to_uint8_image(img_tensor: torch.Tensor) -> np.ndarray:
    """
    img_tensor: [1, 3, H, W] or [3, H, W], identity-normalized, range ~[0,1].
    Returns uint8 RGB HxWx3.
    """
    if img_tensor.ndim == 4:
        img_tensor = img_tensor[0]
    arr = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


def save_rgb(path: Path, img: np.ndarray) -> None:
    Image.fromarray(img).save(path)


def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_int_list(text: str) -> List[int]:
    text = text.strip().lower()
    if text in {"", "none"}:
        return []
    if text == "all":
        return ["all"]  # sentinel
    return [int(x) for x in text.split(",") if x.strip()]


def make_montage(images: List[np.ndarray], names: Optional[List[str]] = None) -> np.ndarray:
    pil_images = [Image.fromarray(img) for img in images]
    widths = [img.width for img in pil_images]
    heights = [img.height for img in pil_images]
    total_w = sum(widths)
    max_h = max(heights)
    canvas = Image.new("RGB", (total_w, max_h + (24 if names else 0)), (255, 255, 255))
    x = 0
    draw = ImageDraw.Draw(canvas)
    for idx, img in enumerate(pil_images):
        canvas.paste(img, (x, 24 if names else 0))
        if names:
            draw.text((x + 5, 4), names[idx], fill=(0, 0, 0))
        x += img.width
    return np.array(canvas)


# -----------------------------------------------------------------------------
# LightGlue sparse feature extraction and pairwise matching
# -----------------------------------------------------------------------------
def build_lightglue_pipeline(
    detector_name: str,
    device: str = "cuda",
    max_num_keypoints: int = 2048,
    lg_filter_threshold: float = 0.1,
    lg_depth_confidence: float = 0.95,
    lg_width_confidence: float = 0.99,
):
    detector_name = detector_name.lower()

    if detector_name == "superpoint":
        extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
        matcher = LightGlue(
            features="superpoint",
            filter_threshold=lg_filter_threshold,
            depth_confidence=lg_depth_confidence,
            width_confidence=lg_width_confidence,
        ).eval().to(device)
    elif detector_name == "aliked":
        extractor = ALIKED(max_num_keypoints=max_num_keypoints).eval().to(device)
        matcher = LightGlue(
            features="aliked",
            filter_threshold=lg_filter_threshold,
            depth_confidence=lg_depth_confidence,
            width_confidence=lg_width_confidence,
        ).eval().to(device)
    elif detector_name == "disk":
        extractor = DISK(max_num_keypoints=max_num_keypoints).eval().to(device)
        matcher = LightGlue(
            features="disk",
            filter_threshold=lg_filter_threshold,
            depth_confidence=lg_depth_confidence,
            width_confidence=lg_width_confidence,
        ).eval().to(device)
    elif detector_name == "sift":
        extractor = SIFT(max_num_keypoints=max_num_keypoints).eval().to(device)
        matcher = LightGlue(
            features="sift",
            filter_threshold=lg_filter_threshold,
            depth_confidence=lg_depth_confidence,
            width_confidence=lg_width_confidence,
        ).eval().to(device)
    else:
        raise ValueError(f"Unsupported LightGlue detector: {detector_name}")

    return extractor, matcher


def _squeeze_lg_image_tensor(img: torch.Tensor) -> torch.Tensor:
    if img.ndim == 4:
        if img.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {img.shape}")
        img = img[0]
    if img.ndim != 3:
        raise ValueError(f"Expected image tensor [3,H,W], got {img.shape}")
    return img.contiguous()


@torch.no_grad()
def extract_features_lightglue(
    image_tensors: List[torch.Tensor],
    extractor,
) -> List[Dict]:
    features = []

    for idx, img in enumerate(image_tensors):
        img = _squeeze_lg_image_tensor(img)

        # Keep LightGlue keypoint coordinates aligned to the already-resized VGGT input.
        try:
            feats_batched = extractor.extract(img, resize=None)
        except TypeError:
            feats_batched = extractor.extract(img)

        feats = rbd(feats_batched)

        kpts = feats["keypoints"].detach().float().cpu()  # [N,2]
        kp_scores = feats.get("keypoint_scores", None)
        if kp_scores is None:
            kp_scores = torch.ones(kpts.shape[0], dtype=torch.float32)
        else:
            kp_scores = kp_scores.detach().float().cpu()

        simple_kps = [
            SimpleKeyPoint(
                pt=(float(kpts[i, 0].item()), float(kpts[i, 1].item())),
                response=float(kp_scores[i].item()),
            )
            for i in range(kpts.shape[0])
        ]

        features.append(
            {
                "keypoints": simple_kps,
                "keypoints_tensor": kpts,
                "keypoint_scores": kp_scores,
                "lg_feats_batched": feats_batched,
                "lg_feats": feats,
            }
        )

        if kpts.shape[0] == 0:
            print(f"[Warn] No LightGlue keypoints extracted for view {idx}.")

    return features


@torch.no_grad()
def extract_pairwise_matches_lightglue(
    features: List[Dict],
    matcher,
    ransac_thresh: float = 1.5,
    ransac_conf: float = 0.999,
    min_inliers: int = 12,
) -> Tuple[Dict[Tuple[int, int], List[PairMatch]], Dict]:
    pair_matches: Dict[Tuple[int, int], List[PairMatch]] = {}
    summary = {}

    for i, j in combinations(range(len(features)), 2):
        feats_i_b = features[i]["lg_feats_batched"]
        feats_j_b = features[j]["lg_feats_batched"]

        matches01 = matcher({"image0": feats_i_b, "image1": feats_j_b})
        matches01 = rbd(matches01)

        if "matches" not in matches01 or matches01["matches"].numel() == 0:
            pair_matches[(i, j)] = []
            summary[(i, j)] = {"raw": 0, "mutual": 0, "inliers": 0}
            continue

        matches = matches01["matches"].detach().long().cpu()  # [K,2]
        match_scores = matches01.get("scores", None)
        if match_scores is None:
            match_scores = torch.ones(matches.shape[0], dtype=torch.float32)
        else:
            match_scores = match_scores.detach().float().cpu()

        kpts_i = features[i]["keypoints_tensor"]
        kpts_j = features[j]["keypoints_tensor"]
        kp_scores_i = features[i]["keypoint_scores"]
        kp_scores_j = features[j]["keypoint_scores"]

        pts_i = kpts_i[matches[:, 0]].numpy().astype(np.float32)
        pts_j = kpts_j[matches[:, 1]].numpy().astype(np.float32)

        if pts_i.shape[0] < 8:
            pair_matches[(i, j)] = []
            summary[(i, j)] = {
                "raw": int(matches.shape[0]),
                "mutual": int(matches.shape[0]),
                "inliers": 0,
            }
            continue

        Fmat, inlier_mask = cv2.findFundamentalMat(
            pts_i,
            pts_j,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=ransac_thresh,
            confidence=ransac_conf,
            maxIters=10000,
        )

        if Fmat is None or inlier_mask is None:
            pair_matches[(i, j)] = []
            summary[(i, j)] = {
                "raw": int(matches.shape[0]),
                "mutual": int(matches.shape[0]),
                "inliers": 0,
            }
            continue

        inlier_mask = inlier_mask.reshape(-1).astype(bool)
        inliers: List[PairMatch] = []

        if int(inlier_mask.sum()) >= min_inliers:
            for keep, (idx_i, idx_j), mscore in zip(inlier_mask, matches.numpy(), match_scores.numpy()):
                if not keep:
                    continue

                xy_i = kpts_i[idx_i]
                xy_j = kpts_j[idx_j]
                s_i = float(kp_scores_i[idx_i].item())
                s_j = float(kp_scores_j[idx_j].item())
                mscore = float(mscore)

                confidence = (
                    0.55 * max(0.0, min(1.0, mscore))
                    + 0.225 * math.tanh(max(s_i, 1e-6))
                    + 0.225 * math.tanh(max(s_j, 1e-6))
                )

                inliers.append(
                    PairMatch(
                        view_a=i,
                        view_b=j,
                        kp_idx_a=int(idx_i),
                        kp_idx_b=int(idx_j),
                        xy_a=(float(xy_i[0]), float(xy_i[1])),
                        xy_b=(float(xy_j[0]), float(xy_j[1])),
                        confidence=float(confidence),
                        distance=0.0,
                        ratio_score=1.0,
                    )
                )

        pair_matches[(i, j)] = inliers
        summary[(i, j)] = {
            "raw": int(matches.shape[0]),
            "mutual": int(matches.shape[0]),
            "inliers": len(inliers),
        }

    return pair_matches, summary


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        self.add(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def build_tracks(
    pair_matches: Dict[Tuple[int, int], List[PairMatch]],
    features: List[Dict],
    min_views_per_track: int = 2,
) -> List[Track]:
    uf = UnionFind()

    for _, matches in pair_matches.items():
        for m in matches:
            a = (m.view_a, m.kp_idx_a)
            b = (m.view_b, m.kp_idx_b)
            uf.union(a, b)

    groups = defaultdict(list)
    for node in uf.parent.keys():
        groups[uf.find(node)].append(node)

    tracks = []
    next_track_id = 0
    for nodes in groups.values():
        per_view = defaultdict(list)
        for view_idx, kp_idx in nodes:
            kp = features[view_idx]["keypoints"][kp_idx]
            per_view[view_idx].append((kp_idx, kp))

        observations = []
        for view_idx, candidates in per_view.items():
            best_kp_idx, best_kp = sorted(
                candidates, key=lambda x: float(x[1].response), reverse=True
            )[0]
            observations.append(
                TrackObservation(
                    view_idx=view_idx,
                    kp_idx=best_kp_idx,
                    xy=(float(best_kp.pt[0]), float(best_kp.pt[1])),
                    response=float(best_kp.response),
                )
            )

        if len(observations) < min_views_per_track:
            continue

        obs_sorted = sorted(observations, key=lambda x: x.view_idx)
        mean_response = float(np.mean([max(o.response, 1e-6) for o in obs_sorted]))
        length_score = min(1.0, len(obs_sorted) / 4.0)
        confidence = 0.6 * length_score + 0.4 * math.tanh(mean_response)

        tracks.append(
            Track(
                track_id=next_track_id,
                observations=obs_sorted,
                confidence=float(confidence),
            )
        )
        next_track_id += 1

    tracks.sort(key=lambda t: (len(t.observations), t.confidence), reverse=True)
    return tracks


# -----------------------------------------------------------------------------
# Pixel -> token mapping and tie-point prior construction
# -----------------------------------------------------------------------------
def pixel_to_patch(
    x: float,
    y: float,
    patch_size: int,
    grid_w: int,
    grid_h: int,
) -> Tuple[int, int, int]:
    px = int(np.clip(np.floor(x / patch_size), 0, grid_w - 1))
    py = int(np.clip(np.floor(y / patch_size), 0, grid_h - 1))
    return px, py, py * grid_w + px


def patch_center_xy(
    patch_linear_idx: int,
    patch_size: int,
    grid_w: int,
) -> Tuple[float, float]:
    py = patch_linear_idx // grid_w
    px = patch_linear_idx % grid_w
    return (px + 0.5) * patch_size, (py + 0.5) * patch_size


def patch_bbox_xyxy(
    patch_linear_idx: int,
    patch_size: int,
    grid_w: int,
) -> Tuple[float, float, float, float]:
    py = patch_linear_idx // grid_w
    px = patch_linear_idx % grid_w
    x0 = px * patch_size
    y0 = py * patch_size
    x1 = x0 + patch_size
    y1 = y0 + patch_size
    return x0, y0, x1, y1


def _gaussian_offsets(radius: int, sigma: float):
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            weight = math.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma + 1e-8))
            offsets.append((dx, dy, weight))
    return offsets


def build_tiepoint_attention_bias(
    tracks: List[Track],
    num_views: int,
    image_h: int,
    image_w: int,
    patch_size: int,
    patch_start_idx: int,
    per_view_tokens: int,
    sigma_q: float = 0.8,
    sigma_k: float = 0.8,
    max_tracks: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Build a flattened global-attention bias of shape [1, 1, S*P, S*P].
    Only cross-view patch-patch entries are biased; camera/register tokens remain zero.
    """
    grid_h = image_h // patch_size
    grid_w = image_w // patch_size
    total_tokens = num_views * per_view_tokens
    bias = torch.zeros((1, 1, total_tokens, total_tokens), dtype=torch.float32)

    radius_q = max(1, int(math.ceil(3.0 * sigma_q)))
    radius_k = max(1, int(math.ceil(3.0 * sigma_k)))
    offsets_q = _gaussian_offsets(radius_q, sigma_q)
    offsets_k = _gaussian_offsets(radius_k, sigma_k)

    used_tracks = tracks if max_tracks is None else tracks[:max_tracks]
    pair_counter = defaultdict(int)

    for track in used_tracks:
        obs = track.observations
        track_len_score = min(1.0, len(obs) / 5.0)
        for oa, ob in combinations(obs, 2):
            for q_obs, k_obs in ((oa, ob), (ob, oa)):
                qx, qy, _ = pixel_to_patch(
                    q_obs.xy[0], q_obs.xy[1], patch_size, grid_w, grid_h
                )
                kx, ky, _ = pixel_to_patch(
                    k_obs.xy[0], k_obs.xy[1], patch_size, grid_w, grid_h
                )

                r = (
                    0.45 * track.confidence
                    + 0.20 * track_len_score
                    + 0.175 * math.tanh(max(q_obs.response, 1e-6))
                    + 0.175 * math.tanh(max(k_obs.response, 1e-6))
                )

                q_offsets = []
                for dx, dy, w in offsets_q:
                    xx = qx + dx
                    yy = qy + dy
                    if 0 <= xx < grid_w and 0 <= yy < grid_h:
                        q_offsets.append((yy * grid_w + xx, w))

                k_offsets = []
                for dx, dy, w in offsets_k:
                    xx = kx + dx
                    yy = ky + dy
                    if 0 <= xx < grid_w and 0 <= yy < grid_h:
                        k_offsets.append((yy * grid_w + xx, w))

                q_base = q_obs.view_idx * per_view_tokens + patch_start_idx
                k_base = k_obs.view_idx * per_view_tokens + patch_start_idx

                for q_patch_idx, q_w in q_offsets:
                    q_global = q_base + q_patch_idx
                    for k_patch_idx, k_w in k_offsets:
                        k_global = k_base + k_patch_idx
                        bias[0, 0, q_global, k_global] += float(r * q_w * k_w)

                pair_counter[(q_obs.view_idx, k_obs.view_idx)] += 1

    metadata = {
        "num_tracks": len(tracks),
        "used_tracks": len(used_tracks),
        "grid_h": grid_h,
        "grid_w": grid_w,
        "patch_size": patch_size,
        "pair_counter": {f"{a}->{b}": c for (a, b), c in pair_counter.items()},
        "bias_max": float(bias.max().item()) if bias.numel() > 0 else 0.0,
        "bias_mean": float(bias.mean().item()) if bias.numel() > 0 else 0.0,
    }
    return bias, metadata


# -----------------------------------------------------------------------------
# Attention patching / capture / bias injection
# -----------------------------------------------------------------------------
class VGGTAttentionController:
    def __init__(
        self,
        model_wrapper: torch.nn.Module,
        capture_layers: Sequence[int],
        inject_layers: Sequence[int],
        head_gains: Optional[Sequence[float]] = None,
        layer_lambda: float = 1.0,
        skip_layers: Optional[Sequence[int]] = None,
    ):
        self.wrapper = model_wrapper
        self.core_model = model_wrapper.model
        self.aggregator = self.core_model.aggregator
        self.capture_layers = list(capture_layers)
        self.inject_layers = list(inject_layers)
        self.skip_layers = sorted(set(skip_layers or []))
        self.layer_lambda = layer_lambda
        self.head_gains = head_gains
        self.saved_attn: Dict[int, torch.Tensor] = {}
        self.saved_logits: Dict[int, torch.Tensor] = {}
        self._patch_global_blocks()

    def _patch_single_attention(self, attn_module: torch.nn.Module, layer_idx: int) -> None:
        attn_module.fused_attn = False
        attn_module._capture_enabled = False
        attn_module._external_bias = None
        attn_module._layer_idx = layer_idx
        attn_module._saved_attn_ref = self.saved_attn
        attn_module._saved_logits_ref = self.saved_logits

        def patched_forward(this, x: torch.Tensor, pos=None) -> torch.Tensor:
            B, N, C = x.shape
            qkv = (
                this.qkv(x)
                .reshape(B, N, 3, this.num_heads, this.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)
            q, k = this.q_norm(q), this.k_norm(k)

            if this.rope is not None:
                q = this.rope(q, pos)
                k = this.rope(k, pos)

            q = q * this.scale
            logits = q @ k.transpose(-2, -1)

            external_bias = getattr(this, "_external_bias", None)
            if external_bias is not None:
                if external_bias.dim() != 4:
                    raise ValueError(
                        f"external_bias must have shape [B or 1, H or 1, N, N], got {external_bias.shape}"
                    )
                # logits = logits + external_bias.to(device=logits.device, dtype=logits.dtype)
                logits = external_bias.to(device=logits.device, dtype=logits.dtype)

            attn = logits.softmax(dim=-1)
            attn = this.attn_drop(attn)
            out = attn @ v
            out = out.transpose(1, 2).reshape(B, N, C)
            out = this.proj(out)
            out = this.proj_drop(out)

            if getattr(this, "_capture_enabled", False):
                this._saved_attn_ref[this._layer_idx] = attn.detach().float().cpu()
                this._saved_logits_ref[this._layer_idx] = logits.detach().float().cpu()

            return out

        attn_module.forward = types.MethodType(patched_forward, attn_module)

    def _patch_skip_block(self, block_module: torch.nn.Module, layer_idx: int) -> None:
        block_module._skip_enabled = True
        block_module._layer_idx = layer_idx

        def skipped_forward(this, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return x

        block_module.forward = types.MethodType(skipped_forward, block_module)

    def _patch_global_blocks(self) -> None:
        for layer_idx, block in enumerate(self.aggregator.global_blocks):
            if layer_idx in self.skip_layers:
                self._patch_skip_block(block, layer_idx)
                continue
            if layer_idx in self.capture_layers or layer_idx in self.inject_layers:
                self._patch_single_attention(block.attn, layer_idx)

    def clear(self) -> None:
        self.saved_attn.clear()
        self.saved_logits.clear()
        for layer_idx, block in enumerate(self.aggregator.global_blocks):
            if layer_idx in self.skip_layers:
                continue
            if hasattr(block.attn, "_capture_enabled"):
                block.attn._capture_enabled = False
            if hasattr(block.attn, "_external_bias"):
                block.attn._external_bias = None

    def enable_capture(self, enabled: bool = True) -> None:
        for layer_idx in self.capture_layers:
            if layer_idx in self.skip_layers:
                continue
            block = self.aggregator.global_blocks[layer_idx]
            if hasattr(block, "attn") and hasattr(block.attn, "_capture_enabled"):
                block.attn._capture_enabled = enabled

    def set_bias(
        self,
        base_bias: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        per_layer_lambdas: Optional[Dict[int, float]] = None,
    ) -> None:
        for layer_idx, block in enumerate(self.aggregator.global_blocks):
            if layer_idx in self.skip_layers:
                continue
            if not hasattr(block.attn, "_external_bias"):
                continue

            if base_bias is None or layer_idx not in self.inject_layers:
                block.attn._external_bias = None
                continue

            lam = self.layer_lambda
            if per_layer_lambdas is not None and layer_idx in per_layer_lambdas:
                lam = per_layer_lambdas[layer_idx]

            bias = base_bias.to(device=device, dtype=dtype)
            if bias.shape[1] == 1:
                if self.head_gains is not None:
                    gains = torch.as_tensor(
                        self.head_gains, dtype=bias.dtype, device=bias.device
                    ).view(1, -1, 1, 1)
                    bias = bias.expand(1, gains.shape[1], bias.shape[-2], bias.shape[-1])
                    bias = bias * gains
                else:
                    bias = bias.expand(1, block.attn.num_heads, bias.shape[-2], bias.shape[-1])
            elif self.head_gains is not None:
                gains = torch.as_tensor(
                    self.head_gains, dtype=bias.dtype, device=bias.device
                ).view(1, -1, 1, 1)
                bias = bias * gains

            block.attn._external_bias = bias * lam

    def get_attention(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self.saved_attn.get(layer_idx, None)

    def get_logits(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self.saved_logits.get(layer_idx, None)


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------
def save_view_pair_match_viz(
    img_a: np.ndarray,
    img_b: np.ndarray,
    matches: List[PairMatch],
    output_path: Path,
    max_draw: int = 200,
) -> None:
    if not matches:
        return
    matches = sorted(matches, key=lambda m: m.confidence, reverse=True)[:max_draw]
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    canvas = Image.new("RGB", (wa + wb, max(ha, hb)), (255, 255, 255))
    canvas.paste(Image.fromarray(img_a), (0, 0))
    canvas.paste(Image.fromarray(img_b), (wa, 0))
    draw = ImageDraw.Draw(canvas)

    for m in matches:
        xa, ya = m.xy_a
        xb, yb = m.xy_b
        score = float(np.clip(m.confidence, 0.0, 1.0))
        color = (
            int(255 * (1.0 - score)),
            int(50 + 180 * score),
            int(255 * score),
        )
        draw.line([(xa, ya), (wa + xb, yb)], fill=color, width=1)
        draw.ellipse((xa - 2, ya - 2, xa + 2, ya + 2), fill=color)
        draw.ellipse((wa + xb - 2, yb - 2, wa + xb + 2, yb + 2), fill=color)

    canvas.save(output_path)


def _prepare_attention_tensor(attn: torch.Tensor) -> torch.Tensor:
    if attn.ndim == 4:
        attn = attn[0]
    if attn.ndim != 3:
        raise ValueError(f"Unexpected attention shape: {attn.shape}")
    return attn.float().cpu()


def _prepare_bias_tensor(bias: torch.Tensor) -> torch.Tensor:
    """
    bias: [1,1,N,N] or [1,H,N,N] or [H,N,N]
    return: [H,N,N]
    """
    if bias.ndim == 4:
        bias = bias[0]
    if bias.ndim != 3:
        raise ValueError(f"Unexpected bias shape: {bias.shape}")
    return bias.detach().float().cpu()


def _extract_pair_attention_block(
    attn: torch.Tensor,
    view_a: int,
    view_b: int,
    per_view_tokens: int,
    patch_start_idx: int,
) -> torch.Tensor:
    attn = _prepare_attention_tensor(attn)
    qa = slice(view_a * per_view_tokens + patch_start_idx, (view_a + 1) * per_view_tokens)
    kb = slice(view_b * per_view_tokens + patch_start_idx, (view_b + 1) * per_view_tokens)
    return attn[:, qa, kb]


def _extract_pair_bias_block(
    bias: torch.Tensor,
    view_a: int,
    view_b: int,
    per_view_tokens: int,
    patch_start_idx: int,
) -> torch.Tensor:
    bias = _prepare_bias_tensor(bias)
    qa = slice(view_a * per_view_tokens + patch_start_idx, (view_a + 1) * per_view_tokens)
    kb = slice(view_b * per_view_tokens + patch_start_idx, (view_b + 1) * per_view_tokens)
    return bias[:, qa, kb]


def _save_heatmap_matrix(
    matrix: np.ndarray,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(matrix, interpolation="nearest")
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_attention_view_heatmap(
    attn: torch.Tensor,
    output_path: Path,
    num_views: int,
    per_view_tokens: int,
    patch_start_idx: int,
    title: str,
) -> None:
    attn = _prepare_attention_tensor(attn).numpy()
    matrix = np.zeros((num_views, num_views), dtype=np.float32)

    for va in range(num_views):
        qa = slice(va * per_view_tokens + patch_start_idx, (va + 1) * per_view_tokens)
        for vb in range(num_views):
            kb = slice(vb * per_view_tokens + patch_start_idx, (vb + 1) * per_view_tokens)
            block = attn[:, qa, kb]
            matrix[va, vb] = block.mean()

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest")
    plt.colorbar()
    plt.xlabel("key view")
    plt.ylabel("query view")
    plt.title(title)
    for i in range(num_views):
        for j in range(num_views):
            plt.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_bias_view_heatmap(
    bias: torch.Tensor,
    output_path: Path,
    num_views: int,
    per_view_tokens: int,
    patch_start_idx: int,
    title: str,
) -> None:
    bias = _prepare_bias_tensor(bias).numpy()
    matrix = np.zeros((num_views, num_views), dtype=np.float32)

    for va in range(num_views):
        qa = slice(va * per_view_tokens + patch_start_idx, (va + 1) * per_view_tokens)
        for vb in range(num_views):
            kb = slice(vb * per_view_tokens + patch_start_idx, (vb + 1) * per_view_tokens)
            block = bias[:, qa, kb]
            matrix[va, vb] = block.mean()

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest")
    plt.colorbar()
    plt.xlabel("key view")
    plt.ylabel("query view")
    plt.title(title)
    for i in range(num_views):
        for j in range(num_views):
            plt.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_attention_pair_matrix(
    attn: torch.Tensor,
    output_path: Path,
    view_a: int,
    view_b: int,
    per_view_tokens: int,
    patch_start_idx: int,
    title: Optional[str] = None,
    save_per_head: bool = True,
) -> None:
    sub = _extract_pair_attention_block(
        attn=attn,
        view_a=view_a,
        view_b=view_b,
        per_view_tokens=per_view_tokens,
        patch_start_idx=patch_start_idx,
    )
    avg = sub.mean(dim=0).numpy()
    base_title = title or f"Attention block view {view_a} -> view {view_b}"
    _save_heatmap_matrix(
        avg,
        output_path,
        title=f"{base_title} (avg heads)",
        xlabel=f"key patches in view {view_b}",
        ylabel=f"query patches in view {view_a}",
    )

    if not save_per_head:
        return

    head_dir = ensure_dir(output_path.parent / f"{output_path.stem}_heads")
    for head_idx in range(sub.shape[0]):
        _save_heatmap_matrix(
            sub[head_idx].numpy(),
            head_dir / f"head_{head_idx:02d}.png",
            title=f"{base_title} (head {head_idx})",
            xlabel=f"key patches in view {view_b}",
            ylabel=f"query patches in view {view_a}",
        )


def save_bias_pair_matrix(
    bias: torch.Tensor,
    output_path: Path,
    view_a: int,
    view_b: int,
    per_view_tokens: int,
    patch_start_idx: int,
    title: Optional[str] = None,
    save_per_head: bool = True,
) -> None:
    sub = _extract_pair_bias_block(
        bias=bias,
        view_a=view_a,
        view_b=view_b,
        per_view_tokens=per_view_tokens,
        patch_start_idx=patch_start_idx,
    )

    avg = sub.mean(dim=0).numpy()
    base_title = title or f"Bias block view {view_a} -> view {view_b}"
    _save_heatmap_matrix(
        avg,
        output_path,
        title=f"{base_title} (avg heads)",
        xlabel=f"key patches in view {view_b}",
        ylabel=f"query patches in view {view_a}",
    )

    if not save_per_head:
        return

    head_dir = ensure_dir(output_path.parent / f"{output_path.stem}_heads")
    for head_idx in range(sub.shape[0]):
        _save_heatmap_matrix(
            sub[head_idx].numpy(),
            head_dir / f"head_{head_idx:02d}.png",
            title=f"{base_title} (head {head_idx})",
            xlabel=f"key patches in view {view_b}",
            ylabel=f"query patches in view {view_a}",
        )


def _save_single_patch_match_viz(
    sub: np.ndarray,
    img_a: np.ndarray,
    img_b: np.ndarray,
    output_path: Path,
    patch_size: int,
    grid_w: int,
    topk: int = 120,
    title: Optional[str] = None,
) -> None:
    best_k = sub.argmax(axis=1)
    best_score = sub.max(axis=1)
    order = np.argsort(best_score)[::-1][:topk]

    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    header_h = 24 if title else 0
    canvas = Image.new("RGB", (wa + wb, max(ha, hb) + header_h), (255, 255, 255))
    canvas.paste(Image.fromarray(img_a), (0, header_h))
    canvas.paste(Image.fromarray(img_b), (wa, header_h))
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((6, 4), title, fill=(0, 0, 0))

    denom = float(best_score.max()) + 1e-8
    for idx in order:
        src_x, src_y = patch_center_xy(int(idx), patch_size, grid_w)
        dst_x, dst_y = patch_center_xy(int(best_k[idx]), patch_size, grid_w)
        score = float(best_score[idx]) / denom
        color = (
            int(255 * score),
            int(64 + 128 * (1.0 - score)),
            int(255 * (1.0 - score)),
        )
        draw.line(
            [(src_x, header_h + src_y), (wa + dst_x, header_h + dst_y)],
            fill=color,
            width=1,
        )
        draw.ellipse(
            (src_x - 2, header_h + src_y - 2, src_x + 2, header_h + src_y + 2),
            fill=color,
        )
        draw.ellipse(
            (wa + dst_x - 2, header_h + dst_y - 2, wa + dst_x + 2, header_h + dst_y + 2),
            fill=color,
        )

    canvas.save(output_path)


def save_attention_patch_match_viz(
    attn: torch.Tensor,
    img_a: np.ndarray,
    img_b: np.ndarray,
    output_path: Path,
    view_a: int,
    view_b: int,
    per_view_tokens: int,
    patch_start_idx: int,
    patch_size: int,
    grid_w: int,
    topk: int = 120,
    title: Optional[str] = None,
    save_per_head: bool = True,
) -> None:
    sub = _extract_pair_attention_block(
        attn=attn,
        view_a=view_a,
        view_b=view_b,
        per_view_tokens=per_view_tokens,
        patch_start_idx=patch_start_idx,
    )

    avg = sub.mean(dim=0).numpy()
    base_title = title or f"Patch matches view {view_a} -> view {view_b}"
    _save_single_patch_match_viz(
        sub=avg,
        img_a=img_a,
        img_b=img_b,
        output_path=output_path,
        patch_size=patch_size,
        grid_w=grid_w,
        topk=topk,
        title=f"{base_title} (avg heads)",
    )

    if not save_per_head:
        return

    head_dir = ensure_dir(output_path.parent / f"{output_path.stem}_heads")
    for head_idx in range(sub.shape[0]):
        _save_single_patch_match_viz(
            sub=sub[head_idx].numpy(),
            img_a=img_a,
            img_b=img_b,
            output_path=head_dir / f"head_{head_idx:02d}.png",
            patch_size=patch_size,
            grid_w=grid_w,
            topk=topk,
            title=f"{base_title} (head {head_idx})",
        )


def _build_regular_grid_samples(
    image_h: int,
    image_w: int,
    sample_rows: int,
    sample_cols: int,
    patch_size: int,
    grid_w: int,
    grid_h: int,
    dedup_tokens: bool = False,
) -> List[Dict]:
    if sample_rows <= 0 or sample_cols <= 0:
        return []

    xs = np.linspace(0.0, float(image_w), sample_cols + 1)
    ys = np.linspace(0.0, float(image_h), sample_rows + 1)

    samples: List[Dict] = []
    seen_tokens = set()
    for row_idx in range(sample_rows):
        for col_idx in range(sample_cols):
            x0 = float(xs[col_idx])
            x1 = float(xs[col_idx + 1])
            y0 = float(ys[row_idx])
            y1 = float(ys[row_idx + 1])
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)

            patch_x, patch_y, patch_idx = pixel_to_patch(
                x=cx,
                y=cy,
                patch_size=patch_size,
                grid_w=grid_w,
                grid_h=grid_h,
            )
            if dedup_tokens and patch_idx in seen_tokens:
                continue
            seen_tokens.add(patch_idx)

            samples.append(
                {
                    "grid_row": int(row_idx),
                    "grid_col": int(col_idx),
                    "cell_bbox_xyxy": [x0, y0, x1, y1],
                    "query_xy": [cx, cy],
                    "query_patch_xy": [int(patch_x), int(patch_y)],
                    "query_patch_idx": int(patch_idx),
                }
            )
    return samples


def _save_single_grid_token_match_viz(
    sub: np.ndarray,
    img_a: np.ndarray,
    img_b: np.ndarray,
    output_path: Path,
    patch_size: int,
    grid_w: int,
    grid_h: int,
    sample_rows: int = 8,
    sample_cols: int = 8,
    title: Optional[str] = None,
    dedup_tokens: bool = False,
) -> None:
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    if ha != hb or wa != wb:
        raise ValueError(f"Expected resized views with identical shape, got {img_a.shape} vs {img_b.shape}")

    samples = _build_regular_grid_samples(
        image_h=ha,
        image_w=wa,
        sample_rows=sample_rows,
        sample_cols=sample_cols,
        patch_size=patch_size,
        grid_w=grid_w,
        grid_h=grid_h,
        dedup_tokens=dedup_tokens,
    )
    if not samples:
        return

    all_scores = []
    match_records = []
    for sample in samples:
        qidx = int(sample["query_patch_idx"])
        row = sub[qidx]
        kidx = int(row.argmax())
        score = float(row[kidx])

        dst_px = int(kidx % grid_w)
        dst_py = int(kidx // grid_w)
        dst_center = patch_center_xy(kidx, patch_size, grid_w)

        record = dict(sample)
        record["matched_patch_idx"] = int(kidx)
        record["matched_patch_xy"] = [dst_px, dst_py]
        record["matched_xy"] = [float(dst_center[0]), float(dst_center[1])]
        record["attention_score"] = score
        match_records.append(record)
        all_scores.append(score)

    score_min = float(min(all_scores))
    score_max = float(max(all_scores))
    score_span = max(score_max - score_min, 1e-8)

    header_h = 24 if title else 0
    canvas = Image.new("RGB", (wa + wb, max(ha, hb) + header_h), (255, 255, 255))
    canvas.paste(Image.fromarray(img_a), (0, header_h))
    canvas.paste(Image.fromarray(img_b), (wa, header_h))
    draw = ImageDraw.Draw(canvas)
    # if title:
    #     draw.text((6, 4), title, fill=(0, 0, 0))

    # # Draw regular grid on the query image for readability.
    # for r in range(1, sample_rows):
    #     y = header_h + int(round(r * ha / sample_rows))
    #     draw.line([(0, y), (wa, y)], fill=(180, 180, 180), width=1)
    # for c in range(1, sample_cols):
    #     x = int(round(c * wa / sample_cols))
    #     draw.line([(x, header_h), (x, header_h + ha)], fill=(180, 180, 180), width=1)

    for record in match_records:
        src_x, src_y = record["query_xy"]
        dst_x, dst_y = record["matched_xy"]
        qidx = int(record["query_patch_idx"])
        kidx = int(record["matched_patch_idx"])
        score = float(record["attention_score"])
        score01 = (score - score_min) / score_span

        color = (
            int(255 * score01),
            int(60 + 160 * (1.0 - score01)),
            int(255 * (1.0 - score01)),
        )

        # qx0, qy0, qx1, qy1 = patch_bbox_xyxy(qidx, patch_size, grid_w)
        # kx0, ky0, kx1, ky1 = patch_bbox_xyxy(kidx, patch_size, grid_w)

        # draw.rectangle((qx0, header_h + qy0, qx1, header_h + qy1), outline=color, width=1)
        # draw.rectangle((wa + kx0, header_h + ky0, wa + kx1, header_h + ky1), outline=color, width=1)
        draw.line(
            [(src_x, header_h + src_y), (wa + dst_x, header_h + dst_y)],
            fill=color,
            width=1,
        )
        draw.ellipse((src_x - 2, header_h + src_y - 2, src_x + 2, header_h + src_y + 2), fill=color)
        draw.ellipse((wa + dst_x - 2, header_h + dst_y - 2, wa + dst_x + 2, header_h + dst_y + 2), fill=color)

    canvas.save(output_path)

    meta = {
        "query_image_size": [int(ha), int(wa)],
        "patch_grid_size": [int(grid_h), int(grid_w)],
        "patch_size": int(patch_size),
        "sample_grid_size": [int(sample_rows), int(sample_cols)],
        "dedup_tokens": bool(dedup_tokens),
        "num_samples": int(len(match_records)),
        "matches": match_records,
    }
    save_json(output_path.with_suffix(".json"), meta)


def save_attention_grid_token_match_viz(
    attn: torch.Tensor,
    img_a: np.ndarray,
    img_b: np.ndarray,
    output_path: Path,
    view_a: int,
    view_b: int,
    per_view_tokens: int,
    patch_start_idx: int,
    patch_size: int,
    grid_w: int,
    grid_h: int,
    sample_rows: int = 8,
    sample_cols: int = 8,
    title: Optional[str] = None,
    save_per_head: bool = False,
    dedup_tokens: bool = False,
) -> None:
    sub = _extract_pair_attention_block(
        attn=attn,
        view_a=view_a,
        view_b=view_b,
        per_view_tokens=per_view_tokens,
        patch_start_idx=patch_start_idx,
    )

    avg = sub.mean(dim=0).numpy()
    base_title = title or f"Grid token matches view {view_a} -> view {view_b}"
    _save_single_grid_token_match_viz(
        sub=avg,
        img_a=img_a,
        img_b=img_b,
        output_path=output_path,
        patch_size=patch_size,
        grid_w=grid_w,
        grid_h=grid_h,
        sample_rows=sample_rows,
        sample_cols=sample_cols,
        title=f"{base_title} (avg heads)",
        dedup_tokens=dedup_tokens,
    )

    if not save_per_head:
        return

    head_dir = ensure_dir(output_path.parent / f"{output_path.stem}_heads")
    for head_idx in range(sub.shape[0]):
        _save_single_grid_token_match_viz(
            sub=sub[head_idx].numpy(),
            img_a=img_a,
            img_b=img_b,
            output_path=head_dir / f"head_{head_idx:02d}.png",
            patch_size=patch_size,
            grid_w=grid_w,
            grid_h=grid_h,
            sample_rows=sample_rows,
            sample_cols=sample_cols,
            title=f"{base_title} (head {head_idx})",
            dedup_tokens=dedup_tokens,
        )


def save_bias_patch_match_viz(
    bias: torch.Tensor,
    img_a: np.ndarray,
    img_b: np.ndarray,
    output_path: Path,
    view_a: int,
    view_b: int,
    per_view_tokens: int,
    patch_start_idx: int,
    patch_size: int,
    grid_w: int,
    topk: int = 120,
    title: Optional[str] = None,
    save_per_head: bool = True,
) -> None:
    sub = _extract_pair_bias_block(
        bias=bias,
        view_a=view_a,
        view_b=view_b,
        per_view_tokens=per_view_tokens,
        patch_start_idx=patch_start_idx,
    )

    avg = sub.mean(dim=0).numpy()
    base_title = title or f"Bias patch matches view {view_a} -> view {view_b}"
    _save_single_patch_match_viz(
        sub=avg,
        img_a=img_a,
        img_b=img_b,
        output_path=output_path,
        patch_size=patch_size,
        grid_w=grid_w,
        topk=topk,
        title=f"{base_title} (avg heads)",
    )

    if not save_per_head:
        return

    head_dir = ensure_dir(output_path.parent / f"{output_path.stem}_heads")
    for head_idx in range(sub.shape[0]):
        _save_single_patch_match_viz(
            sub=sub[head_idx].numpy(),
            img_a=img_a,
            img_b=img_b,
            output_path=head_dir / f"head_{head_idx:02d}.png",
            patch_size=patch_size,
            grid_w=grid_w,
            topk=topk,
            title=f"{base_title} (head {head_idx})",
        )


def save_delta_pair_matrix(
    tensor_after: torch.Tensor,
    tensor_before: torch.Tensor,
    output_path: Path,
    view_a: int,
    view_b: int,
    per_view_tokens: int,
    patch_start_idx: int,
    title: Optional[str] = None,
    save_per_head: bool = True,
    is_attention: bool = True,
) -> None:
    """
    tensor_after / tensor_before: [1,H,N,N] or [H,N,N]
    Save (after - before) for the selected pair block.
    """
    after = _prepare_attention_tensor(tensor_after)
    before = _prepare_attention_tensor(tensor_before)

    qa = slice(view_a * per_view_tokens + patch_start_idx, (view_a + 1) * per_view_tokens)
    kb = slice(view_b * per_view_tokens + patch_start_idx, (view_b + 1) * per_view_tokens)

    sub = after[:, qa, kb] - before[:, qa, kb]
    avg = sub.mean(dim=0).numpy()

    kind = "Δattn" if is_attention else "Δlogits"
    base_title = title or f"{kind} view {view_a} -> view {view_b}"

    plt.figure(figsize=(6.5, 5.5))
    vmax = np.max(np.abs(avg)) + 1e-8
    plt.imshow(avg, interpolation="nearest", cmap="bwr", vmin=-vmax, vmax=vmax)
    plt.colorbar()
    plt.xlabel(f"key patches in view {view_b}")
    plt.ylabel(f"query patches in view {view_a}")
    plt.title(f"{base_title} (avg heads)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    if not save_per_head:
        return

    head_dir = ensure_dir(output_path.parent / f"{output_path.stem}_heads")
    for head_idx in range(sub.shape[0]):
        mat = sub[head_idx].numpy()
        vmax = np.max(np.abs(mat)) + 1e-8
        plt.figure(figsize=(6.5, 5.5))
        plt.imshow(mat, interpolation="nearest", cmap="bwr", vmin=-vmax, vmax=vmax)
        plt.colorbar()
        plt.xlabel(f"key patches in view {view_b}")
        plt.ylabel(f"query patches in view {view_a}")
        plt.title(f"{base_title} (head {head_idx})")
        plt.tight_layout()
        plt.savefig(head_dir / f"head_{head_idx:02d}.png", dpi=200)
        plt.close()


def save_tracks_json(tracks: List[Track], output_path: Path) -> None:
    data = []
    for t in tracks:
        data.append(
            {
                "track_id": t.track_id,
                "confidence": t.confidence,
                "length": len(t.observations),
                "observations": [
                    {
                        "view_idx": o.view_idx,
                        "kp_idx": o.kp_idx,
                        "xy": [o.xy[0], o.xy[1]],
                        "response": o.response,
                    }
                    for o in t.observations
                ],
            }
        )
    save_json(output_path, data)


def _normalize_for_vis(x: np.ndarray) -> np.ndarray:
    mask = np.isfinite(x)
    if not np.any(mask):
        return np.zeros_like(x, dtype=np.float32)
    vals = x[mask]
    lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def _prediction_depth_z(pred: Dict[str, torch.Tensor]) -> np.ndarray:
    if "depth_z" in pred:
        return pred["depth_z"][0, ..., 0].detach().float().cpu().numpy()
    if "pts3d_cam" in pred:
        return pred["pts3d_cam"][0, ..., 2].detach().float().cpu().numpy()
    if "depth_along_ray" in pred and "ray_directions" in pred:
        depth_along_ray = pred["depth_along_ray"][0, ..., 0].detach().float().cpu().numpy()
        ray_dirs = pred["ray_directions"][0].detach().float().cpu().numpy()
        return depth_along_ray * ray_dirs[..., 2]
    raise KeyError("Prediction does not contain depth_z, pts3d_cam, or compatible depth_along_ray/ray_directions fields.")


def _prediction_rgb(pred: Dict[str, torch.Tensor], fallback_rgb: Optional[np.ndarray] = None) -> np.ndarray:
    if "img_no_norm" in pred:
        img = pred["img_no_norm"][0].detach().float().cpu().numpy()
        if img.max() <= 1.5:
            img = img * 255.0
        return np.clip(img, 0, 255).astype(np.uint8)
    if fallback_rgb is not None:
        return fallback_rgb.astype(np.uint8)
    h, w = _prediction_depth_z(pred).shape
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _prediction_mask(pred: Dict[str, torch.Tensor], shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if "mask" in pred:
        mask = pred["mask"][0]
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        return mask.detach().cpu().numpy().astype(bool)
    return np.ones((h, w), dtype=bool)


def _write_ply_binary(path: Path, xyz: np.ndarray, rgb: Optional[np.ndarray] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be [N,3], got {xyz.shape}")

    if rgb is None:
        rgb = np.full((xyz.shape[0], 3), 255, dtype=np.uint8)
    else:
        rgb = np.asarray(rgb)
        if rgb.max() <= 1.5:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if rgb.ndim != 2 or rgb.shape[1] != 3:
            raise ValueError(f"rgb must be [N,3], got {rgb.shape}")
        if rgb.shape[0] != xyz.shape[0]:
            raise ValueError(f"xyz/rgb row mismatch: {xyz.shape[0]} vs {rgb.shape[0]}")

    vertex = np.empty(
        xyz.shape[0],
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex["x"] = xyz[:, 0]
    vertex["y"] = xyz[:, 1]
    vertex["z"] = xyz[:, 2]
    vertex["red"] = rgb[:, 0]
    vertex["green"] = rgb[:, 1]
    vertex["blue"] = rgb[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {xyz.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        vertex.tofile(f)


def save_depth_maps(
    predictions: List[Dict[str, torch.Tensor]],
    output_dir: Path,
    prefix: str,
    fallback_rgbs: Optional[List[np.ndarray]] = None,
) -> None:
    """
    Save depth_z visualizations/raw arrays and fused/per-view PLY point clouds.
    """
    ensure_dir(output_dir)
    merged_xyz = []
    merged_rgb = []

    for idx, pred in enumerate(predictions):
        depth_z = _prediction_depth_z(pred)
        conf = pred["conf"][0].detach().float().cpu().numpy() if "conf" in pred else np.ones_like(depth_z)

        plt.figure(figsize=(6, 5))
        plt.imshow(_normalize_for_vis(depth_z))
        plt.colorbar()
        plt.title(f"{prefix} depth_z view {idx}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_depthz_view{idx:02d}.png", dpi=180)
        plt.close()
        np.save(output_dir / f"{prefix}_depthz_view{idx:02d}.npy", depth_z.astype(np.float32))

        plt.figure(figsize=(6, 5))
        plt.imshow(_normalize_for_vis(conf))
        plt.colorbar()
        plt.title(f"{prefix} conf view {idx}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_conf_view{idx:02d}.png", dpi=180)
        plt.close()
        np.save(output_dir / f"{prefix}_conf_view{idx:02d}.npy", conf.astype(np.float32))

        pts3d_world = None
        if "pts3d" in pred:
            pts3d_world = pred["pts3d"][0].detach().float().cpu().numpy()
        elif "pts3d_cam" in pred and "camera_poses" in pred:
            pts3d_cam = pred["pts3d_cam"][0].detach().float().cpu().numpy()
            pose = pred["camera_poses"][0].detach().float().cpu().numpy()
            pts_h = np.concatenate([pts3d_cam, np.ones_like(pts3d_cam[..., :1])], axis=-1)
            pts3d_world = np.einsum("ij,hwj->hwi", pose, pts_h)[..., :3]

        if pts3d_world is not None:
            rgb = _prediction_rgb(pred, fallback_rgbs[idx] if fallback_rgbs is not None and idx < len(fallback_rgbs) else None)
            valid_mask = _prediction_mask(pred, depth_z.shape)
            valid_mask &= np.isfinite(depth_z)
            valid_mask &= np.isfinite(pts3d_world).all(axis=-1)
            valid_mask &= depth_z > 0

            xyz = pts3d_world[valid_mask].reshape(-1, 3)
            colors = rgb[valid_mask].reshape(-1, 3)
            if xyz.shape[0] > 0:
                _write_ply_binary(output_dir / f"{prefix}_points_view{idx:02d}.ply", xyz, colors)
                merged_xyz.append(xyz)
                merged_rgb.append(colors)

    if merged_xyz:
        merged_xyz = np.concatenate(merged_xyz, axis=0)
        merged_rgb = np.concatenate(merged_rgb, axis=0)
        _write_ply_binary(output_dir / f"{prefix}_points_merged.ply", merged_xyz, merged_rgb)


# -----------------------------------------------------------------------------
# Defaults for capture / injection
# -----------------------------------------------------------------------------
def choose_capture_layers(num_layers: int, spec: Sequence[int]) -> List[int]:
    if spec == ["all"]:
        return list(range(num_layers))
    return [x for x in spec if 0 <= x < num_layers]


def choose_middle_layers(num_layers: int, width: Optional[int] = None) -> List[int]:
    if width is None:
        width = max(4, num_layers // 3)
    center = num_layers // 2
    start = max(0, center - width // 2)
    end = min(num_layers, start + width)
    return list(range(start, end))


def triangular_layer_lambda(layers: Sequence[int], peak: float) -> Dict[int, float]:
    if not layers:
        return {}
    if len(layers) == 1:
        return {layers[0]: peak}
    mid = 0.5 * (layers[0] + layers[-1])
    radius = max(1e-6, 0.5 * (layers[-1] - layers[0]))
    out = {}
    for l in layers:
        score = 1.0 - abs(l - mid) / radius
        score = max(0.25, score)
        out[l] = float(peak * score)
    return out


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VGGT visualization + LightGlue tie-point prior injection")
    parser.add_argument("--image_folder", type=str, default="experiments/test_data/sample4", help="Input image folder")
    parser.add_argument("--output_dir", type=str, default="experiments/test_vggt43", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device; GPU strongly recommended")
    parser.add_argument("--machine", type=str, default="default", help="Hydra machine config")
    parser.add_argument("--model_name", type=str, default="vggt", help="Model config name")
    parser.add_argument("--resolution_set", type=int, default=518, choices=[504, 512, 518])
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--stride", type=int, default=1, help="Load every nth image")
    parser.add_argument("--max_views", type=int, default=0, help="Optional cap on number of views after loading")
    parser.add_argument("--torch_hub_dir", type=str, default="/opt/data/private/code/map-anything/checkpoints/torch_cache/hub")
    parser.add_argument("--local_dino_repo", type=str, default="/opt/data/private/code/map-anything/checkpoints/torch_cache/hub/facebookresearch_dinov2_main")

    parser.add_argument(
        "--detector",
        type=str,
        default="superpoint",
        choices=["superpoint", "aliked", "disk", "sift"],
        help="Local feature extractor used with LightGlue",
    )
    parser.add_argument("--max_num_keypoints", type=int, default=2048)
    parser.add_argument("--lg_filter_threshold", type=float, default=0.1)
    parser.add_argument("--lg_depth_confidence", type=float, default=0.95)
    parser.add_argument("--lg_width_confidence", type=float, default=0.99)
    parser.add_argument("--ransac_thresh", type=float, default=2.0)
    parser.add_argument("--min_inliers", type=int, default=12)
    parser.add_argument("--min_views_per_track", type=int, default=2)
    parser.add_argument("--max_tracks_for_bias", type=int, default=1500)

    parser.add_argument("--capture_layers", type=str, default="")
    parser.add_argument("--inject_layers", type=str, default="11, 13, 15", help="'middle' or comma separated global layer ids")
    parser.add_argument("--bias_lambda", type=float, default=2.0)
    parser.add_argument("--sigma_q", type=float, default=0.8, help="Gaussian spread in query patch grid")
    parser.add_argument("--sigma_k", type=float, default=0.8, help="Gaussian spread in key patch grid")
    parser.add_argument("--head_gains", type=str, default="", help="Comma-separated per-head gains; empty -> all ones")
    # parser.add_argument("--skip_global_layers", type=str, default="2,3,4,5,6,7,8", help="Comma-separated global layer ids to skip entirely during forward")
    parser.add_argument("--skip_global_layers", type=str, default="22,23,21,20", help="Comma-separated global layer ids to skip entirely during forward")


    # Keep the user's original CLI semantics.
    parser.add_argument("--run_baseline", action="store_false", help="Run baseline inference with attention capture")
    parser.add_argument("--run_injected", action="store_true", help="Run inference with tie-point prior injection")
    parser.add_argument("--save_depth", action="store_false", help="Save depth/confidence visualizations")
    parser.add_argument("--save_matches", action="store_false", help="Save pairwise match visualizations")
    parser.add_argument("--pair_a", type=int, default=0, help="Query view index for patch-match visualization")
    parser.add_argument("--pair_b", type=int, default=1, help="Key view index for patch-match visualization")
    parser.add_argument("--topk_patch_matches", type=int, default=120)
    parser.add_argument("--grid_match_rows", type=int, default=16, help="Regular sampling grid rows for token correspondence viz; <=0 disables")
    parser.add_argument("--grid_match_cols", type=int, default=16, help="Regular sampling grid cols for token correspondence viz; <=0 disables")
    parser.add_argument("--grid_match_save_per_head", action="store_true", help="Also save one grid-token match visualization per attention head")
    parser.add_argument("--grid_match_dedup_tokens", action="store_true", help="When regular samples fall into the same token, keep only one sample per token")

    return parser


def main():
    args = build_argparser().parse_args()

    if args.device != "cuda":
        raise RuntimeError(
            "This script is intended for VGGT GPU inference. "
            "The repository wrapper also assumes CUDA for dtype selection."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --device=cuda was requested.")

    output_dir = ensure_dir(Path(args.output_dir))
    viz_dir = ensure_dir(output_dir / "viz")
    match_dir = ensure_dir(viz_dir / "classical_matches")
    attn_dir = ensure_dir(viz_dir / "attention")
    inj_attn_dir = ensure_dir(viz_dir / "attention_injected")
    bias_dir = ensure_dir(viz_dir / "bias")
    compare_dir = ensure_dir(viz_dir / "compare")

    setup_offline_torch_hub(args.torch_hub_dir, args.local_dino_repo)

    print("[Step] Loading model...")
    model = init_model_from_config(args.model_name, device=args.device, machine=args.machine)
    model.eval()

    print("[Step] Loading / preprocessing images...")
    image_paths = list_images(args.image_folder)
    if args.stride > 1:
        image_paths = image_paths[:: args.stride]
    if args.max_views > 0:
        image_paths = image_paths[: args.max_views]

    views = load_images(
        folder_or_list=image_paths,
        resolution_set=args.resolution_set,
        norm_type="identity",
        patch_size=args.patch_size,
        stride=1,
    )
    if len(views) < 2:
        raise RuntimeError("Need at least 2 views.")

    for v in views:
        v["img"] = v["img"].to(args.device)

    resized_rgbs = [tensor_to_uint8_image(v["img"]) for v in views]
    image_names = [Path(p).name for p in image_paths[: len(views)]]
    save_rgb(viz_dir / "inputs_montage.png", make_montage(resized_rgbs, image_names))

    H, W = resized_rgbs[0].shape[:2]
    num_views = len(views)
    aggregator = model.model.aggregator
    patch_size = int(aggregator.patch_size)
    patch_start_idx = int(aggregator.patch_start_idx)
    grid_h = H // patch_size
    grid_w = W // patch_size
    patch_tokens_per_view = grid_h * grid_w
    per_view_tokens = patch_start_idx + patch_tokens_per_view
    num_global_layers = len(aggregator.global_blocks)

    print(
        f"[Info] views={num_views}, image_size=({H},{W}), patch={patch_size}, "
        f"grid=({grid_h},{grid_w}), per_view_tokens={per_view_tokens}, "
        f"global_layers={num_global_layers}"
    )

    capture_layers_spec = parse_int_list(args.capture_layers)
    capture_layers = choose_capture_layers(num_global_layers, capture_layers_spec)

    if args.inject_layers.strip().lower() == "middle":
        inject_layers = choose_middle_layers(num_global_layers)
    else:
        inject_layers = choose_capture_layers(num_global_layers, parse_int_list(args.inject_layers))

    head_gains = None
    if args.head_gains.strip():
        head_gains = [float(x) for x in args.head_gains.split(",") if x.strip()]
        num_heads = int(model.model.aggregator.global_blocks[0].attn.num_heads)
        if len(head_gains) != num_heads:
            raise ValueError(
                f"--head_gains length must equal num_heads={num_heads}, got {len(head_gains)}"
            )

    skip_layers_spec = parse_int_list(args.skip_global_layers)
    skip_layers = choose_capture_layers(num_global_layers, skip_layers_spec)

    effective_capture_layers = [x for x in capture_layers if x not in skip_layers]
    effective_inject_layers = [x for x in inject_layers if x not in skip_layers]

    skipped_from_capture = sorted(set(capture_layers) & set(skip_layers))
    skipped_from_inject = sorted(set(inject_layers) & set(skip_layers))

    print(f"[Info] capture_layers={capture_layers}")
    print(f"[Info] inject_layers={inject_layers}")
    print(f"[Info] skip_global_layers={skip_layers}")
    if skipped_from_capture:
        print(f"[Warn] These capture layers are skipped and will not be visualized: {skipped_from_capture}")
    if skipped_from_inject:
        print(f"[Warn] These inject layers are skipped and will not receive bias: {skipped_from_inject}")

    controller = VGGTAttentionController(
        model_wrapper=model,
        capture_layers=effective_capture_layers,
        inject_layers=effective_inject_layers,
        head_gains=head_gains,
        layer_lambda=args.bias_lambda,
        skip_layers=skip_layers,
    )

    # ------------------------------------------------------------------
    # Step 1: LightGlue sparse matches and multi-view tracks
    # ------------------------------------------------------------------
    print("[Step] Building LightGlue extractor / matcher...")
    extractor, lg_matcher = build_lightglue_pipeline(
        detector_name=args.detector,
        device=args.device,
        max_num_keypoints=args.max_num_keypoints,
        lg_filter_threshold=args.lg_filter_threshold,
        lg_depth_confidence=args.lg_depth_confidence,
        lg_width_confidence=args.lg_width_confidence,
    )

    print("[Step] Extracting LightGlue features...")
    image_tensors_for_lg = [v["img"] for v in views]
    features = extract_features_lightglue(
        image_tensors=image_tensors_for_lg,
        extractor=extractor,
    )

    print("[Step] Pairwise LightGlue matching + RANSAC filtering...")
    pair_matches, pair_summary = extract_pairwise_matches_lightglue(
        features=features,
        matcher=lg_matcher,
        ransac_thresh=args.ransac_thresh,
        min_inliers=args.min_inliers,
    )
    detector_name = f"{args.detector}+lightglue"

    tracks = build_tracks(
        pair_matches=pair_matches,
        features=features,
        min_views_per_track=args.min_views_per_track,
    )

    save_json(
        output_dir / "pair_match_summary.json",
        {f"{a}-{b}": v for (a, b), v in pair_summary.items()},
    )
    save_tracks_json(tracks, output_dir / "tracks.json")

    print(f"[Info] detector={detector_name}, tracks={len(tracks)}")
    top_tracks_info = [
        {
            "track_id": t.track_id,
            "length": len(t.observations),
            "confidence": t.confidence,
        }
        for t in tracks[:20]
    ]
    save_json(output_dir / "top_tracks_summary.json", top_tracks_info)

    if args.save_matches:
        for (a, b), matches in pair_matches.items():
            if not matches:
                continue
            save_view_pair_match_viz(
                resized_rgbs[a],
                resized_rgbs[b],
                matches,
                match_dir / f"pair_{a:02d}_{b:02d}.png",
                max_draw=200,
            )

    # ------------------------------------------------------------------
    # Step 2: baseline inference + attention capture
    # ------------------------------------------------------------------
    baseline_predictions = None
    baseline_attn_cache: Dict[int, torch.Tensor] = {}
    baseline_logits_cache: Dict[int, torch.Tensor] = {}

    if args.run_baseline or (not args.run_baseline and not args.run_injected):
        print("[Step] Running baseline inference...")
        controller.clear()
        controller.enable_capture(True)
        controller.set_bias(None, device=torch.device(args.device))

        with torch.no_grad():
            baseline_predictions = model(views)

        baseline_attn_cache = {k: v.clone() for k, v in controller.saved_attn.items()}
        baseline_logits_cache = {k: v.clone() for k, v in controller.saved_logits.items()}

        if args.save_depth:
            print("[Info] Saving depth...")
            save_depth_maps(baseline_predictions, viz_dir / "baseline_depth", "baseline", fallback_rgbs=resized_rgbs)

        for layer_idx in capture_layers:
            attn = controller.get_attention(layer_idx)
            if attn is None:
                continue
            save_attention_view_heatmap(
                attn=attn,
                output_path=attn_dir / f"layer_{layer_idx:02d}_view_heatmap.png",
                num_views=num_views,
                per_view_tokens=per_view_tokens,
                patch_start_idx=patch_start_idx,
                title=f"Baseline global layer {layer_idx} view-view attention",
            )
            if 0 <= args.pair_a < num_views and 0 <= args.pair_b < num_views and args.pair_a != args.pair_b:
                save_attention_pair_matrix(
                    attn=attn,
                    output_path=attn_dir / f"layer_{layer_idx:02d}_pair_{args.pair_a}_{args.pair_b}_matrix.png",
                    view_a=args.pair_a,
                    view_b=args.pair_b,
                    per_view_tokens=per_view_tokens,
                    patch_start_idx=patch_start_idx,
                    title=f"Baseline layer {layer_idx}: attention block view {args.pair_a} -> view {args.pair_b}",
                    save_per_head=True,
                )
                save_attention_patch_match_viz(
                    attn=attn,
                    img_a=resized_rgbs[args.pair_a],
                    img_b=resized_rgbs[args.pair_b],
                    output_path=attn_dir / f"layer_{layer_idx:02d}_pair_{args.pair_a}_{args.pair_b}.png",
                    view_a=args.pair_a,
                    view_b=args.pair_b,
                    per_view_tokens=per_view_tokens,
                    patch_start_idx=patch_start_idx,
                    patch_size=patch_size,
                    grid_w=grid_w,
                    topk=args.topk_patch_matches,
                    title=f"Baseline layer {layer_idx}: view {args.pair_a} -> view {args.pair_b}",
                    save_per_head=True,
                )
                if args.grid_match_rows > 0 and args.grid_match_cols > 0:
                    save_attention_grid_token_match_viz(
                        attn=attn,
                        img_a=resized_rgbs[args.pair_a],
                        img_b=resized_rgbs[args.pair_b],
                        output_path=attn_dir / f"layer_{layer_idx:02d}_pair_{args.pair_a}_{args.pair_b}_grid_token_match.png",
                        view_a=args.pair_a,
                        view_b=args.pair_b,
                        per_view_tokens=per_view_tokens,
                        patch_start_idx=patch_start_idx,
                        patch_size=patch_size,
                        grid_w=grid_w,
                        grid_h=grid_h,
                        sample_rows=args.grid_match_rows,
                        sample_cols=args.grid_match_cols,
                        title=f"Baseline layer {layer_idx}: grid token matches {args.pair_a} -> {args.pair_b}",
                        save_per_head=args.grid_match_save_per_head,
                        dedup_tokens=args.grid_match_dedup_tokens,
                    )

    # ------------------------------------------------------------------
    # Step 3: build tie-point prior matrix and inject into middle layers
    # ------------------------------------------------------------------
    if args.run_injected:
        print("[Step] Building tie-point attention bias...")
        base_bias, bias_meta = build_tiepoint_attention_bias(
            tracks=tracks,
            num_views=num_views,
            image_h=H,
            image_w=W,
            patch_size=patch_size,
            patch_start_idx=patch_start_idx,
            per_view_tokens=per_view_tokens,
            sigma_q=args.sigma_q,
            sigma_k=args.sigma_k,
            max_tracks=args.max_tracks_for_bias if args.max_tracks_for_bias > 0 else None,
        )
        save_json(output_dir / "bias_metadata.json", bias_meta)

        # ----------------------------------------------------------
        # Visualize the bias itself
        # ----------------------------------------------------------
        save_bias_view_heatmap(
            bias=base_bias,
            output_path=bias_dir / "bias_view_heatmap.png",
            num_views=num_views,
            per_view_tokens=per_view_tokens,
            patch_start_idx=patch_start_idx,
            title="Tie-point bias view-view heatmap",
        )

        if 0 <= args.pair_a < num_views and 0 <= args.pair_b < num_views and args.pair_a != args.pair_b:
            save_bias_pair_matrix(
                bias=base_bias,
                output_path=bias_dir / f"bias_pair_{args.pair_a}_{args.pair_b}_matrix.png",
                view_a=args.pair_a,
                view_b=args.pair_b,
                per_view_tokens=per_view_tokens,
                patch_start_idx=patch_start_idx,
                title=f"Bias block view {args.pair_a} -> view {args.pair_b}",
                save_per_head=True,
            )
            save_bias_patch_match_viz(
                bias=base_bias,
                img_a=resized_rgbs[args.pair_a],
                img_b=resized_rgbs[args.pair_b],
                output_path=bias_dir / f"bias_pair_{args.pair_a}_{args.pair_b}.png",
                view_a=args.pair_a,
                view_b=args.pair_b,
                per_view_tokens=per_view_tokens,
                patch_start_idx=patch_start_idx,
                patch_size=patch_size,
                grid_w=grid_w,
                topk=args.topk_patch_matches,
                title=f"Bias patch matches view {args.pair_a} -> view {args.pair_b}",
                save_per_head=True,
            )

        per_layer_lambdas = triangular_layer_lambda(inject_layers, args.bias_lambda)
        print(f"[Info] per_layer_lambdas={per_layer_lambdas}")

        controller.clear()
        controller.enable_capture(True)
        controller.set_bias(
            base_bias=base_bias,
            device=torch.device(args.device),
            dtype=torch.float32,
            per_layer_lambdas=per_layer_lambdas,
        )

        print("[Step] Running injected inference...")
        with torch.no_grad():
            injected_predictions = model(views)

        if args.save_depth:
            save_depth_maps(injected_predictions, viz_dir / "injected_depth", "injected", fallback_rgbs=resized_rgbs)

        for layer_idx in capture_layers:
            attn = controller.get_attention(layer_idx)
            if attn is None:
                continue
            save_attention_view_heatmap(
                attn=attn,
                output_path=inj_attn_dir / f"layer_{layer_idx:02d}_view_heatmap.png",
                num_views=num_views,
                per_view_tokens=per_view_tokens,
                patch_start_idx=patch_start_idx,
                title=f"Injected global layer {layer_idx} view-view attention",
            )
            if 0 <= args.pair_a < num_views and 0 <= args.pair_b < num_views and args.pair_a != args.pair_b:
                save_attention_pair_matrix(
                    attn=attn,
                    output_path=inj_attn_dir / f"layer_{layer_idx:02d}_pair_{args.pair_a}_{args.pair_b}_matrix.png",
                    view_a=args.pair_a,
                    view_b=args.pair_b,
                    per_view_tokens=per_view_tokens,
                    patch_start_idx=patch_start_idx,
                    title=f"Injected layer {layer_idx}: attention block view {args.pair_a} -> view {args.pair_b}",
                    save_per_head=True,
                )
                save_attention_patch_match_viz(
                    attn=attn,
                    img_a=resized_rgbs[args.pair_a],
                    img_b=resized_rgbs[args.pair_b],
                    output_path=inj_attn_dir / f"layer_{layer_idx:02d}_pair_{args.pair_a}_{args.pair_b}.png",
                    view_a=args.pair_a,
                    view_b=args.pair_b,
                    per_view_tokens=per_view_tokens,
                    patch_start_idx=patch_start_idx,
                    patch_size=patch_size,
                    grid_w=grid_w,
                    topk=args.topk_patch_matches,
                    title=f"Injected layer {layer_idx}: view {args.pair_a} -> view {args.pair_b}",
                    save_per_head=True,
                )
                if args.grid_match_rows > 0 and args.grid_match_cols > 0:
                    save_attention_grid_token_match_viz(
                        attn=attn,
                        img_a=resized_rgbs[args.pair_a],
                        img_b=resized_rgbs[args.pair_b],
                        output_path=inj_attn_dir / f"layer_{layer_idx:02d}_pair_{args.pair_a}_{args.pair_b}_grid_token_match.png",
                        view_a=args.pair_a,
                        view_b=args.pair_b,
                        per_view_tokens=per_view_tokens,
                        patch_start_idx=patch_start_idx,
                        patch_size=patch_size,
                        grid_w=grid_w,
                        grid_h=grid_h,
                        sample_rows=args.grid_match_rows,
                        sample_cols=args.grid_match_cols,
                        title=f"Injected layer {layer_idx}: grid token matches {args.pair_a} -> {args.pair_b}",
                        save_per_head=args.grid_match_save_per_head,
                        dedup_tokens=args.grid_match_dedup_tokens,
                    )

        # ----------------------------------------------------------
        # Compare baseline vs injected: did bias actually change logits/attn?
        # ----------------------------------------------------------
        if baseline_attn_cache and baseline_logits_cache:
            for layer_idx in capture_layers:
                if layer_idx not in baseline_attn_cache:
                    continue
                if layer_idx not in controller.saved_attn:
                    continue
                if layer_idx not in baseline_logits_cache:
                    continue
                if layer_idx not in controller.saved_logits:
                    continue

                inj_attn = controller.saved_attn[layer_idx]
                inj_logits = controller.saved_logits[layer_idx]
                base_attn = baseline_attn_cache[layer_idx]
                base_logits = baseline_logits_cache[layer_idx]

                if 0 <= args.pair_a < num_views and 0 <= args.pair_b < num_views and args.pair_a != args.pair_b:
                    save_delta_pair_matrix(
                        tensor_after=inj_logits,
                        tensor_before=base_logits,
                        output_path=compare_dir / f"layer_{layer_idx:02d}_pair_{args.pair_a}_{args.pair_b}_delta_logits.png",
                        view_a=args.pair_a,
                        view_b=args.pair_b,
                        per_view_tokens=per_view_tokens,
                        patch_start_idx=patch_start_idx,
                        title=f"Layer {layer_idx}: injected - baseline logits",
                        save_per_head=True,
                        is_attention=False,
                    )

                    save_delta_pair_matrix(
                        tensor_after=inj_attn,
                        tensor_before=base_attn,
                        output_path=compare_dir / f"layer_{layer_idx:02d}_pair_{args.pair_a}_{args.pair_b}_delta_attn.png",
                        view_a=args.pair_a,
                        view_b=args.pair_b,
                        per_view_tokens=per_view_tokens,
                        patch_start_idx=patch_start_idx,
                        title=f"Layer {layer_idx}: injected - baseline attention",
                        save_per_head=True,
                        is_attention=True,
                    )
        else:
            print("[Warn] Baseline cache is empty; skipping delta_logits / delta_attn comparison.")

    summary = {
        "num_views": num_views,
        "image_size": [H, W],
        "patch_size": patch_size,
        "grid_size": [grid_h, grid_w],
        "per_view_tokens": per_view_tokens,
        "num_tracks": len(tracks),
        "capture_layers": capture_layers,
        "inject_layers": inject_layers,
        "skip_global_layers": skip_layers,
        "effective_capture_layers": effective_capture_layers,
        "effective_inject_layers": effective_inject_layers,
        "pair_a": args.pair_a,
        "pair_b": args.pair_b,
        "matcher": detector_name,
    }
    save_json(output_dir / "run_summary.json", summary)

    print("[Done] Outputs written to:", output_dir)


if __name__ == "__main__":
    main()
