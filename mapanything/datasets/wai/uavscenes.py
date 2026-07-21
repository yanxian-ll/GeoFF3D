"""
UAVScenes Dataset using WAI format data.
"""

import os
import json

import torch
import cv2
import numpy as np

from mapanything.datasets.base.base_dataset import BaseDataset
from mapanything.datasets.wai.a3dreal import A3DRealWAI

class UAVScenesWAI(A3DRealWAI):
    """
    UAVScenes dataset containing object-centric and birds-eye-view scenes.
    """

    def __init__(
        self,
        *args,
        ROOT,
        dataset_metadata_dir,
        split,
        overfit_num_sets=None,
        sample_specific_scene: bool = False,
        specific_scene_name: str = None,
        load_modalities: list = ["image", "depth"],
        covisibility_thres_max: float = 1.0,
        sampling_mode: str = "random_walk",
        walk_restart_prob: float = 0.10,
        walk_temperature: float = 1.0,
        walk_topk_step: int = 50,
        # hfov-balanced scene sampling
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
            use_hfov_balanced_sampling=use_hfov_balanced_sampling,
            **kwargs
        )
        # Indicate synthetic dataset
        self.is_synthetic = False
        self.is_metric_scale = True

    def _load_data(self):
        split_metadata_path = os.path.join(
            self.dataset_metadata_dir,
            self.split,
            f"uavscenes_scene_list_{self.split}.npy",
        )
        split_scene_list = np.load(split_metadata_path, allow_pickle=True)

        if not self.sample_specific_scene:
            self.scenes = list(split_scene_list)
        else:
            self.scenes = [self.specific_scene_name]

        self.num_of_scenes = len(self.scenes)

        if self.use_hfov_balanced_sampling:
            hfov_metadata_path = os.path.join(
                self.dataset_metadata_dir,
                self.split,
                f"uavscenes_scene_hfov_{self.split}.json",
            )
            self._load_hfov_scene_info(hfov_metadata_path)


    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        views = super()._get_views(sampled_idx, num_views_to_sample, resolution)
        for view in views:
            view["dataset"] = "UAVScenes"
        return views
    
def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd", "--root_dir", default="../../dataset/data/uavscenes", type=str
    )
    parser.add_argument(
        "-dmd",
        "--dataset_metadata_dir",
        default="../../dataset/metadata",
        type=str,
    )
    parser.add_argument(
        "-nv",
        "--num_of_views",
        default=16,
        type=int,
    )
    parser.add_argument("--viz", action="store_true", default=False)

    return parser


if __name__ == "__main__":
    import rerun as rr
    from tqdm import tqdm

    from mapanything.datasets.base.base_dataset import view_name
    from mapanything.utils.image import rgb
    from mapanything.utils.viz import script_add_rerun_args

    parser = get_parser()
    script_add_rerun_args(
        parser
    )  # Options: --headless, --connect, --serve, --addr, --save, --stdout
    args = parser.parse_args()

    dataset = UAVScenesWAI(
        num_views=args.num_of_views,
        split="train",
        covisibility_thres=0.1,
        ROOT=args.root_dir,
        dataset_metadata_dir=args.dataset_metadata_dir,
        resolution=(518, 392),
        aug_crop=16,
        transform="colorjitter+grayscale+gaublur",
        data_norm_type="dinov2",
        interval=5,
    )
    print(dataset.get_stats())

    if args.viz:
        rr.script_setup(args, "UAVScenes_Dataloader")
        rr.set_time("stable_time", sequence=0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    sampled_indices = np.random.choice(len(dataset), size=10, replace=False)

    for num, idx in enumerate(tqdm(sampled_indices)):
        views = dataset[idx]
        assert len(views) == args.num_of_views
        sample_name = f"{idx}"
        for view_idx in range(args.num_of_views):
            sample_name += f" {view_name(views[view_idx])}"
        print(sample_name)
        for view_idx in range(args.num_of_views):
            image = rgb(
                views[view_idx]["img"], norm_type=views[view_idx]["data_norm_type"]
            )
            depthmap = views[view_idx]["depthmap"]
            pose = views[view_idx]["camera_pose"]
            intrinsics = views[view_idx]["camera_intrinsics"]
            pts3d = views[view_idx]["pts3d"]
            valid_mask = views[view_idx]["valid_mask"]
            if "non_ambiguous_mask" in views[view_idx]:
                non_ambiguous_mask = views[view_idx]["non_ambiguous_mask"]
            else:
                non_ambiguous_mask = None
            if "prior_depth_along_ray" in views[view_idx]:
                prior_depth_along_ray = views[view_idx]["prior_depth_along_ray"]
            else:
                prior_depth_along_ray = None
            if args.viz:
                rr.set_time("stable_time", sequence=num)
                base_name = f"world/view_{view_idx}"
                pts_name = f"world/view_{view_idx}_pointcloud"
                # Log camera info and loaded data
                height, width = image.shape[0], image.shape[1]
                rr.log(
                    base_name,
                    rr.Transform3D(
                        translation=pose[:3, 3],
                        mat3x3=pose[:3, :3],
                    ),
                )
                rr.log(
                    f"{base_name}/pinhole",
                    rr.Pinhole(
                        image_from_camera=intrinsics,
                        height=height,
                        width=width,
                        camera_xyz=rr.ViewCoordinates.RDF,
                    ),
                )
                rr.log(
                    f"{base_name}/pinhole/rgb",
                    rr.Image(image),
                )
                rr.log(
                    f"{base_name}/pinhole/depth",
                    rr.DepthImage(depthmap),
                )
                if prior_depth_along_ray is not None:
                    rr.log(
                        f"prior_depth_along_ray_{view_idx}",
                        rr.DepthImage(prior_depth_along_ray),
                    )
                if non_ambiguous_mask is not None:
                    rr.log(
                        f"{base_name}/pinhole/non_ambiguous_mask",
                        rr.SegmentationImage(non_ambiguous_mask.astype(int)),
                    )
                # Log points in 3D
                filtered_pts = pts3d[valid_mask]
                filtered_pts_col = image[valid_mask]
                rr.log(
                    pts_name,
                    rr.Points3D(
                        positions=filtered_pts.reshape(-1, 3),
                        colors=filtered_pts_col.reshape(-1, 3),
                    ),
                )

                