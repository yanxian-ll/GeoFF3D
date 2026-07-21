"""
A3D-Syn-L Focal Ambiguity Dataset using WAI format data.
"""

import os
import numpy as np

from mapanything.datasets.wai.a3dsynl import A3DSynLargeWAI


class A3DSynLargeFAWAI(A3DSynLargeWAI):
    """
    A3D-Syn-L dataset containing object-centric and birds-eye-view scenes.
    """

    def _load_data(self):
        split_metadata_path = os.path.join(
            self.dataset_metadata_dir,
            self.split,
            f"UAVFF3D-FA_scene_list_{self.split}.npy",
        )
        split_scene_list = np.load(split_metadata_path, allow_pickle=True)

        if not self.sample_specific_scene:
            self.scenes = list(split_scene_list)
        else:
            self.scenes = [self.specific_scene_name]
        self.num_of_scenes = len(self.scenes)

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        views = super()._get_views(sampled_idx, num_views_to_sample, resolution)
        for view in views:
            view["dataset"] = "UAVFF3D-FA"
        return views


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd", "--root_dir", default="../../dataset/data/A3D-Syn-L", type=str
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
        default=8,
        type=int,
    )
    parser.add_argument(
        "--num_samples",
        default=2,
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
        default="./experiments/dataset_viz/synl_fa_viz.rrd",
        help="Output .rrd file path",
    )
    parser.add_argument(
        "--app_id",
        type=str,
        default="A3D-Syn-L_Dataloader",
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

    dataset = A3DSynLargeFAWAI(
        num_views=args.num_of_views,
        split="test",
        covisibility_thres=0.1,
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

