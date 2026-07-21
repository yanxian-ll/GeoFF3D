# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Utilities for random sampling under a single or multiple constraints

References: DUSt3R
"""

import numpy as np
import torch
import math

def round_by(total, multiple, up=False):
    """
    Round a number to the nearest multiple of another number.

    Args:
        total (int): The number to round
        multiple (int): The multiple to round to
        up (bool, optional): Whether to round up. Defaults to False.

    Returns:
        int: The rounded number
    """
    if up:
        total = total + multiple - 1
    return (total // multiple) * multiple


class BatchedRandomSampler:
    """
    Random sampling under a constraint: each sample in the batch has the same feature,
    which is chosen randomly from a known pool of 'features' for each batch.

    For instance, the 'feature' could be the image aspect-ratio.

    The index returned is a tuple (sample_idx, feat_idx).
    This sampler ensures that each series of `batch_size` indices has the same `feat_idx`.
    """

    def __init__(
        self, dataset, batch_size, pool_size, world_size=1, rank=0, drop_last=True
    ):
        """
        Args:
            dataset: Dataset to sample from
            batch_size: Number of samples per batch
            pool_size: Integer representing the size of feature pool
            world_size: Number of distributed processes
            rank: Rank of the current process
            drop_last: Whether to drop the last incomplete batch
        """
        self.batch_size = batch_size
        self.pool_size = pool_size

        self.len_dataset = N = len(dataset)
        self.total_size = round_by(N, batch_size * world_size) if drop_last else N
        assert world_size == 1 or drop_last, (
            "must drop the last batch in distributed mode"
        )

        # Distributed sampler
        self.world_size = world_size
        self.rank = rank
        self.epoch = None

    def __len__(self):
        """
        Get the length of the sampler.

        Returns:
            int: The number of samples in the sampler for the current process
        """
        return self.total_size // self.world_size

    def set_epoch(self, epoch):
        """
        Set the epoch for this sampler.

        This should be called before each epoch to ensure proper shuffling of the data.

        Args:
            epoch (int): The current epoch number
        """
        self.epoch = epoch

    def __iter__(self):
        """
        Iterator over the indices.

        This method generates random indices for each batch, ensuring that all samples
        within a batch have the same feature index for the given feature pool.

        Yields:
            tuple: A tuple containing (sample_idx, feat_idx)
        """
        # Prepare RNG
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, (
                "use set_epoch() if distributed mode is used"
            )
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        # Random indices (will restart from 0 if not drop_last)
        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)

        # Random feat_idxs (same across each batch)
        n_batches = (self.total_size + self.batch_size - 1) // self.batch_size
        feat_idxs = rng.integers(self.pool_size, size=n_batches)
        feat_idxs = np.broadcast_to(feat_idxs[:, None], (n_batches, self.batch_size))
        feat_idxs = feat_idxs.ravel()[: self.total_size]

        # Put them together
        idxs = np.c_[sample_idxs, feat_idxs]  # shape = (total_size, 2)

        # Distributed sampler: we select a subset of batches
        # Make sure the slice for each node is aligned with batch_size
        size_per_proc = self.batch_size * (
            (self.total_size + self.world_size * self.batch_size - 1)
            // (self.world_size * self.batch_size)
        )
        idxs = idxs[self.rank * size_per_proc : (self.rank + 1) * size_per_proc]

        yield from (tuple(idx) for idx in idxs)


class BatchedMultiFeatureRandomSampler:
    """
    Random sampling under multiple constraints: each sample in the batch has the same features,
    which are chosen randomly from known pools of 'features' for each batch.

    For instance, the 'features' could be the image aspect-ratio and scene type.

    The index returned is a tuple (sample_idx, feat_idx_1, feat_idx_2, ...).
    This sampler ensures that each series of `batch_size` indices has the same feature indices.
    """

    def __init__(
        self, dataset, batch_size, pool_sizes, world_size=1, rank=0, drop_last=True
    ):
        """
        Args:
            dataset: Dataset to sample from
            batch_size: Number of samples per batch
            pool_sizes: List of integers representing the size of each feature pool
            world_size: Number of distributed processes
            rank: Rank of the current process
            drop_last: Whether to drop the last incomplete batch
        """
        self.batch_size = batch_size
        self.pool_sizes = pool_sizes if isinstance(pool_sizes, list) else [pool_sizes]

        self.len_dataset = N = len(dataset)
        self.total_size = round_by(N, batch_size * world_size) if drop_last else N
        assert world_size == 1 or drop_last, (
            "must drop the last batch in distributed mode"
        )

        # Distributed sampler
        self.world_size = world_size
        self.rank = rank
        self.epoch = None

    def __len__(self):
        """
        Get the length of the sampler.

        Returns:
            int: The number of samples in the sampler for the current process
        """
        return self.total_size // self.world_size

    def set_epoch(self, epoch):
        """
        Set the epoch for this sampler.

        This should be called before each epoch to ensure proper shuffling of the data.

        Args:
            epoch (int): The current epoch number
        """
        self.epoch = epoch

    def __iter__(self):
        """
        Iterator over the indices.

        This method generates random indices for each batch, ensuring that all samples
        within a batch have the same feature indices for multiple features.

        Yields:
            tuple: A tuple containing (sample_idx, feat_idx_1, feat_idx_2, ...)
        """
        # Prepare RNG
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, (
                "use set_epoch() if distributed mode is used"
            )
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        # Random indices (will restart from 0 if not drop_last)
        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)

        # Random feat_idxs (same across each batch)
        n_batches = (self.total_size + self.batch_size - 1) // self.batch_size

        # Generate feature indices for each feature pool
        all_feat_idxs = []
        for pool_size in self.pool_sizes:
            feat_idxs = rng.integers(pool_size, size=n_batches)
            feat_idxs = np.broadcast_to(
                feat_idxs[:, None], (n_batches, self.batch_size)
            )
            feat_idxs = feat_idxs.ravel()[: self.total_size]
            all_feat_idxs.append(feat_idxs)

        # Put them together
        idxs = np.column_stack(
            [sample_idxs] + all_feat_idxs
        )  # shape = (total_size, 1 + len(pool_sizes))

        # Distributed sampler: we select a subset of batches
        # Make sure the slice for each node is aligned with batch_size
        size_per_proc = self.batch_size * (
            (self.total_size + self.world_size * self.batch_size - 1)
            // (self.world_size * self.batch_size)
        )
        idxs = idxs[self.rank * size_per_proc : (self.rank + 1) * size_per_proc]

        yield from (tuple(idx) for idx in idxs)


class DynamicBatchedMultiFeatureRandomSampler:
    """
    Random sampling under multiple constraints with dynamic batch size:
    each sample in the batch has the same features, which are chosen randomly
    from known pools of 'features' for each batch.

    The batch size is dynamically determined based on a specified feature index,
    using a direct mapping from feature values to batch sizes.

    For instance, if one of the features is the number of images in a multi-view set,
    you can specify different batch sizes for different numbers of images to optimize
    GPU memory usage. This is achieved by using the feature_to_batch_size_map parameter
    to directly specify what batch size to use for each feature value.

    The returned index is a list of tuples [(sample_idx, feat_idx_1, feat_idx_2, ...), ...].
    """

    def __init__(
        self,
        dataset,
        pool_sizes,
        scaling_feature_idx=0,
        feature_to_batch_size_map=None,
        world_size=1,
        rank=0,
        drop_last=True,
    ):
        """
        Args:
            dataset: Dataset to sample from
            pool_sizes: List of integers representing the size of each feature pool
            scaling_feature_idx: Index of the feature to use for determining batch size (0-based index into pool_sizes)
            feature_to_batch_size_map: Optional function or dict that maps feature values directly to batch sizes.
                                 For example, if the feature represents number of views, this maps number of views
                                 to appropriate batch size that can fit in GPU memory.
                                 If None, uses a default batch size of 1 for all feature values.
            world_size: Number of distributed processes
            rank: Rank of the current process
            drop_last: Whether to drop the last incomplete batch
        """
        self.pool_sizes = pool_sizes if isinstance(pool_sizes, list) else [pool_sizes]
        self.scaling_feature_idx = scaling_feature_idx

        # Ensure scaling_feature_idx is valid
        if scaling_feature_idx < 0 or scaling_feature_idx >= len(self.pool_sizes):
            raise ValueError(
                f"scaling_feature_idx must be between 0 and {len(self.pool_sizes) - 1}"
            )

        # Set up mapping from feature values to batch sizes
        self.feature_to_batch_size_map = feature_to_batch_size_map
        if self.feature_to_batch_size_map is None:
            # Default: batch size of 1 for all feature values
            self.feature_to_batch_size_map = {
                i: 1 for i in range(self.pool_sizes[scaling_feature_idx])
            }

        self.len_dataset = N = len(dataset)

        # We don't know the exact batch size yet, so we use a large number for total_size
        # This will be adjusted during iteration
        self.total_size = N

        # Distributed sampler
        self.world_size = world_size
        self.rank = rank
        self.epoch = None
        self.drop_last = drop_last

    def __len__(self):
        """
        Get the approximate length of the sampler.

        Since batch size varies, this is an estimate based on the largest batch size
        in the mapping, which provides a lower bound on the number of batches.

        Returns:
            int: The estimated minimum number of samples in the sampler for the current process
        """
        # Find the largest batch size in the mapping
        if callable(self.feature_to_batch_size_map):
            # If it's a function, sample some values to find the maximum
            batch_sizes = [
                self.feature_to_batch_size_map(i)
                for i in range(self.pool_sizes[self.scaling_feature_idx])
            ]
            max_batch_size = max(batch_sizes)
        else:
            # If it's a dict or similar, find the maximum directly
            max_batch_size = max(self.feature_to_batch_size_map.values())

        # Ensure minimum batch size of 1
        max_batch_size = max(1, max_batch_size)

        # Estimate total batches using the largest batch size
        # This gives a lower bound on the number of batches
        total_batches = self.total_size // max_batch_size
        if not self.drop_last and self.total_size % max_batch_size > 0:
            total_batches += 1

        # Distribute among processes
        return total_batches // self.world_size

    def set_epoch(self, epoch):
        """
        Set the epoch for this sampler.

        This should be called before each epoch to ensure proper shuffling of the data.

        Args:
            epoch (int): The current epoch number
        """
        self.epoch = epoch

    def __iter__(self):
        """
        Iterator over the indices with dynamic batch sizes.

        This method generates random indices for each batch, ensuring that all samples
        within a batch have the same feature indices for multiple features.
        The batch size is determined directly from the feature_to_batch_size_map.

        The iterator enforces the length returned by __len__() by stopping after
        exactly that many batches have been yielded for this process.

        Yields:
            list of tuples: A batch of tuples, each containing (sample_idx, feat_idx_1, feat_idx_2, ...)
        """
        # Prepare RNG
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, (
                "use set_epoch() if distributed mode is used"
            )
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        # Random indices for the entire dataset
        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)

        # Get the target number of batches for this process (enforce strict length)
        target_batches_for_process = len(self)
        batches_yielded_for_process = 0

        # Process indices in batches with dynamic sizing
        idx = 0
        batch_idx = 0  # Track batch index for even distribution
        while idx < len(sample_idxs) and (
            batches_yielded_for_process < target_batches_for_process
        ):
            # Randomly select feature indices for this batch
            feat_idxs = [rng.integers(pool_size) for pool_size in self.pool_sizes]

            # Get the scaling feature value
            scaling_feat = feat_idxs[self.scaling_feature_idx]

            # Get the batch size directly from the mapping
            if callable(self.feature_to_batch_size_map):
                batch_size = self.feature_to_batch_size_map(scaling_feat)
            else:
                batch_size = self.feature_to_batch_size_map.get(scaling_feat, 1)

            # Ensure minimum batch size of 1
            batch_size = max(1, batch_size)

            # Ensure we don't go beyond available samples
            remaining = len(sample_idxs) - idx
            if remaining < batch_size:
                if self.drop_last:
                    break
                batch_size = remaining

            # Create batch with consistent feature indices
            batch = []
            for i in range(batch_size):
                if idx + i < len(sample_idxs):
                    sample_idx = sample_idxs[idx + i]
                    batch.append(tuple([sample_idx] + feat_idxs))

            # Distribute batches among processes in round-robin fashion
            if len(batch) > 0 and (batch_idx % self.world_size == self.rank):
                yield batch
                batches_yielded_for_process += 1

            batch_idx += 1  # Increment batch index
            idx += batch_size


def _shuffle_np_copy(arr, rng):
    arr = np.asarray(arr, dtype=np.int64).copy()
    rng.shuffle(arr)
    return arr


class HFOVBalancedBatchedMultiFeatureRandomSampler:
    """
    Scene-level hfov-balanced sampler (non-dynamic batch size).

    Each batch is assigned:
      1) one hfov bin (chosen in round-robin / approximately uniform order),
      2) one shared set of feature indices (e.g. aspect ratio, num_views_idx),
      3) scene indices sampled only from that hfov bin.

    Returned item format is identical to the original non-dynamic sampler:
        (sample_idx, feat_idx_1, feat_idx_2, ...)
    so BaseDataset._getitem_fn() does not need to change.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        pool_sizes,
        hfov_bin_to_scene_indices,
        world_size=1,
        rank=0,
        drop_last=True,
    ):
        self.batch_size = int(batch_size)
        self.pool_sizes = pool_sizes if isinstance(pool_sizes, list) else [pool_sizes]

        self.len_dataset = N = len(dataset)
        self.total_size = round_by(N, batch_size * world_size) if drop_last else N

        assert world_size == 1 or drop_last, "must drop the last batch in distributed mode"

        self.world_size = world_size
        self.rank = rank
        self.epoch = None

        self.hfov_bin_to_scene_indices = {
            int(k): np.asarray(v, dtype=np.int64)
            for k, v in hfov_bin_to_scene_indices.items()
            if len(v) > 0
        }
        if len(self.hfov_bin_to_scene_indices) == 0:
            raise ValueError("hfov_bin_to_scene_indices is empty.")

        self.available_hfov_bins = sorted(self.hfov_bin_to_scene_indices.keys())

    def __len__(self):
        return self.total_size // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, (
                "use set_epoch() if distributed mode is used"
            )
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        n_batches = (self.total_size + self.batch_size - 1) // self.batch_size

        # Build near-uniform bin schedule per batch.
        repeats = math.ceil(n_batches / max(1, len(self.available_hfov_bins)))
        tiled = []
        for _ in range(repeats):
            bins = np.array(self.available_hfov_bins, dtype=np.int64)
            rng.shuffle(bins)
            tiled.append(bins)
        bin_schedule = np.concatenate(tiled)[:n_batches]

        # Per-bin shuffled pools + pointers
        bin_pools = {
            b: _shuffle_np_copy(scene_indices, rng)
            for b, scene_indices in self.hfov_bin_to_scene_indices.items()
        }
        bin_ptrs = {b: 0 for b in self.available_hfov_bins}

        all_rows = []
        for batch_idx in range(n_batches):
            cur_bin = int(bin_schedule[batch_idx])

            # shared feature indices for the whole batch
            feat_idxs = [int(rng.integers(pool_size)) for pool_size in self.pool_sizes]

            batch_rows = []
            for _ in range(self.batch_size):
                pool = bin_pools[cur_bin]
                ptr = bin_ptrs[cur_bin]

                if ptr >= len(pool):
                    pool = _shuffle_np_copy(self.hfov_bin_to_scene_indices[cur_bin], rng)
                    bin_pools[cur_bin] = pool
                    bin_ptrs[cur_bin] = 0
                    ptr = 0

                sample_idx = int(pool[ptr])
                bin_ptrs[cur_bin] += 1
                batch_rows.append(tuple([sample_idx] + feat_idxs))

            all_rows.extend(batch_rows)

        all_rows = all_rows[: self.total_size]

        # Distributed slice aligned with batch size
        size_per_proc = self.batch_size * (
            (self.total_size + self.world_size * self.batch_size - 1)
            // (self.world_size * self.batch_size)
        )
        all_rows = all_rows[self.rank * size_per_proc : (self.rank + 1) * size_per_proc]
        yield from all_rows


class HFOVBalancedDynamicBatchedMultiFeatureRandomSampler:
    """
    Scene-level hfov-balanced sampler with dynamic batch size.

    Each yielded item is a batch:
        [(sample_idx, feat_idx_1, feat_idx_2, ...), ...]
    matching the interface expected by DynamicBatchDatasetWrapper.
    """

    def __init__(
        self,
        dataset,
        pool_sizes,
        hfov_bin_to_scene_indices,
        scaling_feature_idx=0,
        feature_to_batch_size_map=None,
        world_size=1,
        rank=0,
        drop_last=True,
    ):
        self.pool_sizes = pool_sizes if isinstance(pool_sizes, list) else [pool_sizes]
        self.scaling_feature_idx = scaling_feature_idx
        if scaling_feature_idx < 0 or scaling_feature_idx >= len(self.pool_sizes):
            raise ValueError(
                f"scaling_feature_idx must be between 0 and {len(self.pool_sizes) - 1}"
            )

        self.feature_to_batch_size_map = feature_to_batch_size_map
        if self.feature_to_batch_size_map is None:
            self.feature_to_batch_size_map = {
                i: 1 for i in range(self.pool_sizes[scaling_feature_idx])
            }

        self.len_dataset = N = len(dataset)
        self.total_size = N

        self.world_size = world_size
        self.rank = rank
        self.epoch = None
        self.drop_last = drop_last

        self.hfov_bin_to_scene_indices = {
            int(k): np.asarray(v, dtype=np.int64)
            for k, v in hfov_bin_to_scene_indices.items()
            if len(v) > 0
        }
        if len(self.hfov_bin_to_scene_indices) == 0:
            raise ValueError("hfov_bin_to_scene_indices is empty.")

        self.available_hfov_bins = sorted(self.hfov_bin_to_scene_indices.keys())

    def __len__(self):
        if callable(self.feature_to_batch_size_map):
            batch_sizes = [
                self.feature_to_batch_size_map(i)
                for i in range(self.pool_sizes[self.scaling_feature_idx])
            ]
            max_batch_size = max(batch_sizes)
        else:
            max_batch_size = max(self.feature_to_batch_size_map.values())

        max_batch_size = max(1, max_batch_size)
        total_batches = self.total_size // max_batch_size
        if not self.drop_last and self.total_size % max_batch_size > 0:
            total_batches += 1
        return total_batches // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, (
                "use set_epoch() if distributed mode is used"
            )
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        target_batches_for_process = len(self)
        total_batches = target_batches_for_process * self.world_size

        # Near-uniform bin schedule
        repeats = math.ceil(total_batches / max(1, len(self.available_hfov_bins)))
        tiled = []
        for _ in range(repeats):
            bins = np.array(self.available_hfov_bins, dtype=np.int64)
            rng.shuffle(bins)
            tiled.append(bins)
        bin_schedule = np.concatenate(tiled)[:total_batches]

        # Per-bin shuffled pools + pointers
        bin_pools = {
            b: _shuffle_np_copy(scene_indices, rng)
            for b, scene_indices in self.hfov_bin_to_scene_indices.items()
        }
        bin_ptrs = {b: 0 for b in self.available_hfov_bins}

        yielded_batches_for_process = 0
        batch_global_idx = 0

        while yielded_batches_for_process < target_batches_for_process:
            cur_bin = int(bin_schedule[batch_global_idx])

            feat_idxs = [int(rng.integers(pool_size)) for pool_size in self.pool_sizes]
            scaling_feat = feat_idxs[self.scaling_feature_idx]

            if callable(self.feature_to_batch_size_map):
                batch_size = self.feature_to_batch_size_map(scaling_feat)
            else:
                batch_size = self.feature_to_batch_size_map.get(scaling_feat, 1)
            batch_size = max(1, int(batch_size))

            batch = []
            for _ in range(batch_size):
                pool = bin_pools[cur_bin]
                ptr = bin_ptrs[cur_bin]

                if ptr >= len(pool):
                    pool = _shuffle_np_copy(self.hfov_bin_to_scene_indices[cur_bin], rng)
                    bin_pools[cur_bin] = pool
                    bin_ptrs[cur_bin] = 0
                    ptr = 0

                sample_idx = int(pool[ptr])
                bin_ptrs[cur_bin] += 1
                batch.append(tuple([sample_idx] + feat_idxs))

            if batch_global_idx % self.world_size == self.rank:
                yield batch
                yielded_batches_for_process += 1

            batch_global_idx += 1
