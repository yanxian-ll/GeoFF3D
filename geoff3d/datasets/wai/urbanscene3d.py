"""UrbanScene3D dataset in WAI format."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from geoff3d.datasets.wai.a3dreal import A3DRealWAI


class UrbanScene3DWAI(A3DRealWAI):
    """Metric-scale UrbanScene3D scenes using the shared WAI loader."""

    def __init__(
        self,
        *args,
        ROOT,
        dataset_metadata_dir,
        split,
        overfit_num_sets=None,
        sample_specific_scene: bool = False,
        specific_scene_name: Optional[str] = None,
        load_modalities: list[str] = ["image", "depth"],
        covisibility_thres_max: float = 1.0,
        sampling_mode: str = "random_walk",
        walk_restart_prob: float = 0.10,
        walk_temperature: float = 1.0,
        walk_topk_step: int = 50,
        tree_branching: int = 2,
        tree_trunk_ratio: float = 0.25,
        mixed_anchor_star_prob: float = 0.50,
        mixed_random_walk_prob: float = 0.25,
        mixed_tree_prob: float = 0.15,
        mixed_greedy_chain_prob: float = 0.10,
        use_hfov_balanced_sampling: bool = False,
        **kwargs,
    ):
        super().__init__(
            *args,
            ROOT=ROOT,
            dataset_metadata_dir=dataset_metadata_dir,
            split=split,
            overfit_num_sets=overfit_num_sets,
            sample_specific_scene=sample_specific_scene,
            specific_scene_name=specific_scene_name,
            load_modalities=load_modalities,
            covisibility_thres_max=covisibility_thres_max,
            sampling_mode=sampling_mode,
            walk_restart_prob=walk_restart_prob,
            walk_temperature=walk_temperature,
            walk_topk_step=walk_topk_step,
            tree_branching=tree_branching,
            tree_trunk_ratio=tree_trunk_ratio,
            mixed_anchor_star_prob=mixed_anchor_star_prob,
            mixed_random_walk_prob=mixed_random_walk_prob,
            mixed_tree_prob=mixed_tree_prob,
            mixed_greedy_chain_prob=mixed_greedy_chain_prob,
            use_hfov_balanced_sampling=use_hfov_balanced_sampling,
            **kwargs,
        )
        self.is_synthetic = False
        self.is_metric_scale = True

    def _load_data(self):
        scene_list_path = os.path.join(
            self.dataset_metadata_dir,
            self.split,
            f"urbanscene3d_scene_list_{self.split}.npy",
        )
        split_scene_list = np.load(scene_list_path, allow_pickle=True)
        self.scenes = (
            [self.specific_scene_name]
            if self.sample_specific_scene
            else list(split_scene_list)
        )
        self.num_of_scenes = len(self.scenes)

        if self.use_hfov_balanced_sampling:
            hfov_path = os.path.join(
                self.dataset_metadata_dir,
                self.split,
                f"urbanscene3d_scene_hfov_{self.split}.json",
            )
            self._load_hfov_scene_info(hfov_path)

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        views = super()._get_views(sampled_idx, num_views_to_sample, resolution)
        for view in views:
            view["dataset"] = "UrbanScene3D"
        return views
