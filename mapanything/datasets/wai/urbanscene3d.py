"""
UrbanScene3D Dataset using WAI format data.
"""

import os
import numpy as np

from mapanything.datasets.wai.a3dreal import A3DRealWAI


class UrbanScene3DWAI(A3DRealWAI):
    """
    UrbanScene3D dataset.
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

        # sampling mode
        sampling_mode: str = "random_walk",   # "anchor_star" | "random_walk" | "tree" | "greedy_chain" | "mixed"

        # random-walk params
        walk_restart_prob: float = 0.10,
        walk_temperature: float = 1.0,
        walk_topk_step: int = 50,

        # tree params
        tree_branching: int = 2,
        tree_trunk_ratio: float = 0.25,

        # mixed sampling probabilities
        mixed_anchor_star_prob: float = 0.50,
        mixed_random_walk_prob: float = 0.25,
        mixed_tree_prob: float = 0.15,
        mixed_greedy_chain_prob: float = 0.10,
        
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
        split_metadata_path = os.path.join(
            self.dataset_metadata_dir,
            self.split,
            f"urbanscene3d_scene_list_{self.split}.npy",
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
                f"urbanscene3d_scene_hfov_{self.split}.json",
            )
            self._load_hfov_scene_info(hfov_metadata_path)

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        views = super()._get_views(sampled_idx, num_views_to_sample, resolution)
        for view in views:
            view["dataset"] = "UrbanScene3D"
        return views



def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd", "--root_dir", default="../../dataset/data/urbanscene3d", type=str
    )
    parser.add_argument(
        "-dmd",
        "--dataset_metadata_dir",
        default="../../dataset/data/metadata",
        type=str,
    )
    parser.add_argument(
        "-nv",
        "--num_of_views",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--num_samples",
        default=3,
        type=int,
        help="How many dataset samples to export into the RRD",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed for sampled indices",
    )
    parser.add_argument(
        "--save_rrd",
        type=str,
        default="./experiments/dataset_viz/urbanscene3d_viz.rrd",
        help="Output .rrd file path",
    )
    parser.add_argument(
        "--app_id",
        type=str,
        default="urbanscene3d_Dataloader",
        help="Rerun application ID",
    )
    return parser


import rerun as rr
from tqdm import tqdm
from mapanything.datasets.base.base_dataset import view_name
from mapanything.utils.image import rgb
from pathlib import Path


def setup_rerun(save_rrd: str, app_id: str):
    save_path = Path(save_rrd)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rr.init(app_id, spawn=False)
    rr.save(str(save_path))
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    return save_path


def log_view_to_rerun(view_idx: int, view: dict):
    image = rgb(view["img"], norm_type=view["data_norm_type"])
    depthmap = view["depthmap"]
    pose = view["camera_pose"]
    intrinsics = view["camera_intrinsics"]
    pts3d = view["pts3d"]
    valid_mask = view["valid_mask"]

    non_ambiguous_mask = view.get("non_ambiguous_mask", None)
    prior_depth_along_ray = view.get("prior_depth_along_ray", None)

    base_name = f"world/view_{view_idx}"
    pts_name = f"world/view_{view_idx}_pointcloud"

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
            f"{base_name}/pinhole/prior_depth_along_ray",
            rr.DepthImage(prior_depth_along_ray),
        )

    if non_ambiguous_mask is not None:
        rr.log(
            f"{base_name}/pinhole/non_ambiguous_mask",
            rr.SegmentationImage(non_ambiguous_mask.astype(np.uint8)),
        )

    filtered_pts = pts3d[valid_mask]
    filtered_pts_col = image[valid_mask]
    rr.log(
        pts_name,
        rr.Points3D(
            positions=filtered_pts.reshape(-1, 3),
            colors=filtered_pts_col.reshape(-1, 3),
        ),
    )


def main():
    parser = get_parser()
    args = parser.parse_args()

    dataset = UrbanScene3DWAI(
        num_views=args.num_of_views,
        split="test",
        covisibility_thres=0.5,
        covisibility_thres_max=1.0,
        ROOT=args.root_dir,
        dataset_metadata_dir=args.dataset_metadata_dir,
        resolution=(518, 392),
        transform="imgnorm",
        data_norm_type="dinov2",
        sampling_mode="random_walk",  # "random_walk" or "greedy_chain"
    )
    print(dataset.get_stats())

    save_path = setup_rerun(args.save_rrd, args.app_id)

    rng = np.random.default_rng(args.seed)
    num_samples = min(args.num_samples, len(dataset))
    sampled_indices = rng.choice(len(dataset), size=num_samples, replace=False)

    for num, idx in enumerate(tqdm(sampled_indices)):
        rr.set_time("stable_time", sequence=num)

        views = dataset[idx]
        assert len(views) == args.num_of_views

        sample_name = f"{idx}"
        for view_idx in range(args.num_of_views):
            sample_name += f" {view_name(views[view_idx])}"
        print(sample_name)

        for view_idx in range(args.num_of_views):
            log_view_to_rerun(view_idx, views[view_idx])

    print(f"[Done] Saved RRD to: {save_path}")
    print("[Next] Copy this .rrd file to your local machine and open it with: rerun <file.rrd>")


if __name__ == "__main__":
    main()




# def get_parser():
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "-rd", "--root_dir", default="../../dataset/data/urbanscene3d", type=str
#     )
#     parser.add_argument(
#         "-dmd",
#         "--dataset_metadata_dir",
#         default="../../dataset/data/metadata",
#         type=str,
#     )
#     parser.add_argument(
#         "-nv",
#         "--num_of_views",
#         default=16,
#         type=int,
#     )
#     parser.add_argument("--viz", action="store_true", default=False)

#     return parser


# # python urbanscene3d.py --viz --serve

# if __name__ == "__main__":
#     import rerun as rr
#     from tqdm import tqdm

#     from mapanything.datasets.base.base_dataset import view_name
#     from mapanything.utils.image import rgb
#     from mapanything.utils.viz import script_add_rerun_args

#     parser = get_parser()
#     script_add_rerun_args(
#         parser
#     )  # Options: --headless, --connect, --serve, --addr, --save, --stdout
#     args = parser.parse_args()

#     dataset = UrbanScene3DWAI(
#         num_views=args.num_of_views,
#         split="test",
#         covisibility_thres=0.1,
#         covisibility_thres_max=1.0,
#         ROOT=args.root_dir,
#         dataset_metadata_dir=args.dataset_metadata_dir,
#         resolution=(518, 392),
#         transform="imgnorm",
#         data_norm_type="dinov2",
#         sampling_mode="random_walk",  # "random_walk" or "greedy_chain"
#     )
#     print(dataset.get_stats())

#     if args.viz:
#         rr.script_setup(args, "UrbanScene3D_Dataloader")
#         rr.set_time("stable_time", sequence=0)
#         rr.log("world", rr.ViewCoordinates.RDF, static=True)

#     sampled_indices = np.random.choice(len(dataset), size=3, replace=False)

#     for num, idx in enumerate(tqdm(sampled_indices)):
#         views = dataset[idx]
#         assert len(views) == args.num_of_views
#         sample_name = f"{idx}"
#         for view_idx in range(args.num_of_views):
#             sample_name += f" {view_name(views[view_idx])}"
#         print(sample_name)
#         for view_idx in range(args.num_of_views):
#             image = rgb(
#                 views[view_idx]["img"], norm_type=views[view_idx]["data_norm_type"]
#             )
#             depthmap = views[view_idx]["depthmap"]
#             pose = views[view_idx]["camera_pose"]
#             intrinsics = views[view_idx]["camera_intrinsics"]
#             pts3d = views[view_idx]["pts3d"]
#             valid_mask = views[view_idx]["valid_mask"]
#             if "non_ambiguous_mask" in views[view_idx]:
#                 non_ambiguous_mask = views[view_idx]["non_ambiguous_mask"]
#             else:
#                 non_ambiguous_mask = None
#             if "prior_depth_along_ray" in views[view_idx]:
#                 prior_depth_along_ray = views[view_idx]["prior_depth_along_ray"]
#             else:
#                 prior_depth_along_ray = None
#             if args.viz:
#                 rr.set_time("stable_time", sequence=num)
#                 base_name = f"world/view_{view_idx}"
#                 pts_name = f"world/view_{view_idx}_pointcloud"
#                 # Log camera info and loaded data
#                 height, width = image.shape[0], image.shape[1]
#                 rr.log(
#                     base_name,
#                     rr.Transform3D(
#                         translation=pose[:3, 3],
#                         mat3x3=pose[:3, :3],
#                     ),
#                 )
#                 rr.log(
#                     f"{base_name}/pinhole",
#                     rr.Pinhole(
#                         image_from_camera=intrinsics,
#                         height=height,
#                         width=width,
#                         camera_xyz=rr.ViewCoordinates.RDF,
#                     ),
#                 )
#                 rr.log(
#                     f"{base_name}/pinhole/rgb",
#                     rr.Image(image),
#                 )
#                 rr.log(
#                     f"{base_name}/pinhole/depth",
#                     rr.DepthImage(depthmap),
#                 )
#                 if prior_depth_along_ray is not None:
#                     rr.log(
#                         f"prior_depth_along_ray_{view_idx}",
#                         rr.DepthImage(prior_depth_along_ray),
#                     )
#                 if non_ambiguous_mask is not None:
#                     rr.log(
#                         f"{base_name}/pinhole/non_ambiguous_mask",
#                         rr.SegmentationImage(non_ambiguous_mask.astype(int)),
#                     )
#                 # Log points in 3D
#                 filtered_pts = pts3d[valid_mask]
#                 filtered_pts_col = image[valid_mask]
#                 rr.log(
#                     pts_name,
#                     rr.Points3D(
#                         positions=filtered_pts.reshape(-1, 3),
#                         colors=filtered_pts_col.reshape(-1, 3),
#                     ),
#                 )

