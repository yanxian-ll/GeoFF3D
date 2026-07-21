# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything Datasets
"""

import torch

from mapanything.datasets.base.base_dataset import BaseDataset
from mapanything.datasets.base.world_frame_augmentation import WorldFrameAugmentDataset  # noqa


_ORIGINAL_BASE_DATASET_INIT = BaseDataset.__init__


def _base_dataset_init_with_min_num_views(
    self,
    num_views: int,
    variable_num_views: bool = False,
    *args,
    min_num_views: int = 2,
    **kwargs,
):
    """Add configurable min_num_views while preserving BaseDataset behavior.

    BaseDataset historically hard-coded variable_num_views to sample from
    [2, num_views]. This wrapper keeps the old default but allows dataset
    configs to pass min_num_views so training can sample [min_num_views,
    num_views] instead.
    """
    min_num_views = int(min_num_views)
    if min_num_views < 1:
        raise ValueError(f"min_num_views must be >= 1, got {min_num_views}")
    if min_num_views > int(num_views):
        raise ValueError(
            f"min_num_views ({min_num_views}) must be <= num_views ({num_views})"
        )

    # Disable the original hard-coded [2, num_views] conversion, then apply the
    # configurable range below.
    _ORIGINAL_BASE_DATASET_INIT(
        self,
        num_views=num_views,
        variable_num_views=False,
        *args,
        **kwargs,
    )
    self.variable_num_views = variable_num_views
    self.num_views_min = min_num_views
    if self.variable_num_views and self.num_views > self.num_views_min:
        self.num_views = list(range(self.num_views_min, self.num_views + 1))


BaseDataset.__init__ = _base_dataset_init_with_min_num_views

from mapanything.datasets.wai.aerialmegadepth import AerialMegaDepthWAI  # noqa
from mapanything.datasets.wai.ase import ASEWAI  # noqa
from mapanything.datasets.wai.blendedmvs import BlendedMVSWAI  # noqa
from mapanything.datasets.wai.dl3dv import DL3DVWAI  # noqa
from mapanything.datasets.wai.dynamicreplica import DynamicReplicaWAI  # noqa
from mapanything.datasets.wai.eth3d import ETH3DWAI  # noqa
from mapanything.datasets.wai.megadepth import MegaDepthWAI  # noqa
from mapanything.datasets.wai.mpsd import MPSDWAI  # noqa
from mapanything.datasets.wai.mvs_synth import MVSSynthWAI  # noqa
from mapanything.datasets.wai.paralleldomain4d import ParallelDomain4DWAI  # noqa
from mapanything.datasets.wai.sailvos3d import SAILVOS3DWAI  # noqa
from mapanything.datasets.wai.scannetpp import ScanNetPPWAI  # noqa
from mapanything.datasets.wai.spring import SpringWAI  # noqa
from mapanything.datasets.wai.tav2_wb import TartanAirV2WBWAI  # noqa
from mapanything.datasets.wai.unrealstereo4k import UnrealStereo4KWAI  # noqa
from mapanything.utils.train_tools import get_rank, get_world_size

from mapanything.datasets.wai.whu_whuomvs import WHUWHUOMVSWAI # noqa
from mapanything.datasets.wai.uavscenes import UAVScenesWAI # noqa
# from mapanything.datasets.wai.ortholoc import OrthoLocWAI # noqa
from mapanything.datasets.wai.a3dreal import A3DRealWAI       # noqa
from mapanything.datasets.wai.a3dsynl import A3DSynLargeWAI   # noqa
from mapanything.datasets.wai.a3dsynl_fa import A3DSynLargeFAWAI   # noqa
from mapanything.datasets.wai.a3dsyns import A3DSynSmallWAI   # noqa
from mapanything.datasets.wai.a3dscenes_instance import A3DScenesWAIInstance # noqa
from mapanything.datasets.wai.a3dscenes_depth_completion import A3DScenesDepthCompletionWAI # noqa
from mapanything.datasets.wai.usegeo import UseGeoWAI # noqa
from mapanything.datasets.wai.urbanscene3d import UrbanScene3DWAI # noqa
from mapanything.datasets.wai.enrich import ENRICHWAI # noqa


def _maybe_wrap_world_frame_augmentation(dataset, world_frame_augmentation=None):
    """Optionally wrap a train dataset with world-frame augmentation."""
    if world_frame_augmentation is None:
        return dataset

    enabled = bool(world_frame_augmentation.get("enabled", False))
    if not enabled:
        return dataset

    return WorldFrameAugmentDataset(
        dataset,
        enabled=enabled,
        rotation_deg=world_frame_augmentation.get("rotation_deg", None),
        x_rotation_deg=world_frame_augmentation.get("x_rotation_deg", None),
        y_rotation_deg=world_frame_augmentation.get("y_rotation_deg", None),
        z_rotation_deg=world_frame_augmentation.get("z_rotation_deg", 0.0),
        translation_range=world_frame_augmentation.get("translation_range", 0.0),
        recenter=world_frame_augmentation.get("recenter", False),
        recenter_mode=world_frame_augmentation.get("recenter_mode", "first_camera"),
        scale_range=world_frame_augmentation.get("scale_range", (0.9, 1.1)),
    )


def get_test_data_loader(
    dataset, batch_size, num_workers=8, shuffle=False, drop_last=False, pin_mem=True
):
    "Get simple PyTorch dataloader corresponding to the testing dataset"
    # PyTorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    if torch.distributed.is_initialized():
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    elif shuffle:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    return data_loader


def get_test_many_ar_data_loader(
    dataset, batch_size, num_workers=8, drop_last=False, pin_mem=True):
    "Get PyTorch dataloader corresponding to the testing dataset that supports many aspect ratios"
    # PyTorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    # Get BatchedMultiFeatureRandomSampler
    sampler = dataset.make_sampler(
        batch_size,
        shuffle=True,
        world_size=world_size,
        rank=rank,
        drop_last=drop_last,
        use_dynamic_sampler=False,
    )

    # Init the data laoder
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    return data_loader


class DynamicBatchDatasetWrapper:
    """
    Wrapper dataset that handles DynamicBatchedMultiFeatureRandomSampler output.

    The dynamic sampler returns batches (lists of tuples) instead of individual samples.
    This wrapper ensures that the underlying dataset's __getitem__ method gets called
    with individual tuples as expected.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, batch_indices):
        """
        Handle batch of indices from DynamicBatchedMultiFeatureRandomSampler.

        Args:
            batch_indices: List of tuples like [(sample_idx, feat_idx_1, feat_idx_2, ...), ...]

        Returns:
            List of samples from the underlying dataset
        """
        if isinstance(batch_indices, (list, tuple)) and len(batch_indices) > 0:
            # If it's a batch (list of tuples), process each item
            if isinstance(batch_indices[0], (list, tuple)):
                return [self.dataset[idx] for idx in batch_indices]
            else:
                # Single tuple, call dataset directly
                return self.dataset[batch_indices]
        else:
            # Fallback for single index
            return self.dataset[batch_indices]

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        # Delegate all other attributes to the wrapped dataset
        return getattr(self.dataset, name)


def get_train_data_loader(
    dataset,
    max_num_of_imgs_per_gpu,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    world_frame_augmentation=None,
):
    "Dynamic PyTorch dataloader corresponding to the training dataset"
    # PyTorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    dataset = _maybe_wrap_world_frame_augmentation(dataset, world_frame_augmentation)

    world_size = get_world_size()
    rank = get_rank()

    # Get DynamicBatchedMultiFeatureRandomSampler
    batch_sampler = dataset.make_sampler(
        shuffle=shuffle,
        world_size=world_size,
        rank=rank,
        drop_last=drop_last,
        max_num_of_images_per_gpu=max_num_of_imgs_per_gpu,
        use_dynamic_sampler=True,
    )

    # Wrap the dataset to handle batch format from dynamic sampler
    wrapped_dataset = DynamicBatchDatasetWrapper(dataset)

    # Init the dynamic data loader
    data_loader = torch.utils.data.DataLoader(
        wrapped_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    return data_loader
