import os
import cv2
import numpy as np
import torch

from mapanything.datasets.base.base_dataset import BaseDataset
from mapanything.utils.wai.core import load_data, load_frame


class A3DScenesWAISequentialSingleView(BaseDataset):
    """
    A3DScenes WAI dataset that iterates ALL frames sequentially (one image per sample).
    Each __getitem__ returns a list with ONE view dict, compatible with BaseDataset processing.
    """

    def __init__(
        self,
        *args,
        ROOT,
        dataset_metadata_dir,
        sample_specific_scene: bool = False,
        specific_scene_name: str = None,
        load_modalities: list = ["image", "depth", "mask"],
        overfit_num_sets=None,  # optional: limit number of returned samples
        **kwargs,
    ):
        # Force single-view behavior
        kwargs = dict(kwargs)
        kwargs["variable_num_views"] = False
        super().__init__(*args, num_views=1, **kwargs)

        self.ROOT = ROOT
        self.dataset_metadata_dir = dataset_metadata_dir
        self.sample_specific_scene = sample_specific_scene
        self.specific_scene_name = specific_scene_name
        self.load_modalities = load_modalities
        self.overfit_num_sets = overfit_num_sets

        # Dataset flags
        self.is_metric_scale = True
        self.is_synthetic = False

        # Build flat index: [(scene_name, frame_name), ...]
        self._build_flat_index()

    def _load_data(self):
        # not used; we build in _build_flat_index
        self.scenes = []
        self.num_of_scenes = 0

    def _build_flat_index(self):
        # train split
        split_metadata_path = os.path.join(
            self.dataset_metadata_dir,
            self.split,
            f"a3dscenes_scene_list_train.npy",
        )
        train_split_scene_list = np.load(split_metadata_path, allow_pickle=True)

        # test split
        split_metadata_path = os.path.join(
            self.dataset_metadata_dir,
            self.split,
            f"a3dscenes_scene_list_test.npy",
        )
        test_split_scene_list = np.load(split_metadata_path, allow_pickle=True)

        # val split
        split_metadata_path = os.path.join(
            self.dataset_metadata_dir,
            self.split,
            f"a3dscenes_scene_list_val.npy",
        )
        val_split_scene_list = np.load(split_metadata_path, allow_pickle=True)

        # all   
        split_scene_list = train_split_scene_list + test_split_scene_list + val_split_scene_list

        if not self.sample_specific_scene:
            scenes = list(split_scene_list)
        else:
            scenes = [self.specific_scene_name]

        # Make order deterministic
        scenes = sorted(scenes)

        flat = []
        self._scene_meta_cache = {}

        for scene_name in scenes:
            scene_root = os.path.join(self.ROOT, scene_name)
            scene_meta = load_data(os.path.join(scene_root, "scene_meta.json"), "scene_meta")
            self._scene_meta_cache[scene_name] = scene_meta

            frame_names = list(scene_meta["frame_names"].keys())
            frame_names = sorted(frame_names)

            for fn in frame_names:
                flat.append((scene_name, fn))

        if self.overfit_num_sets is not None:
            flat = flat[: int(self.overfit_num_sets)]

        self.flat_index = flat
        self.scenes = scenes
        self.num_of_scenes = len(scenes)

    def __len__(self):
        return len(self.flat_index)

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        """
        Return ONE view, sequentially indexed.
        BaseDataset will compute pts3d/valid_mask/rays/etc afterwards.
        """
        scene_name, frame_name = self.flat_index[int(sampled_idx)]
        scene_root = os.path.join(self.ROOT, scene_name)
        scene_meta = self._scene_meta_cache[scene_name]

        view_data = load_frame(
            scene_root,
            frame_name,
            modalities=self.load_modalities,
            scene_meta=scene_meta,
        )

        raw_image = view_data["image"].permute(1, 2, 0).numpy()  # (H,W,3) float [0,1]
        raw_image = (raw_image * 255).astype(np.uint8)

        depthmap = view_data["depth"].numpy().astype(np.float32)
        intrinsics = view_data["intrinsics"].numpy().astype(np.float32)
        c2w_pose = view_data["extrinsics"].numpy().astype(np.float32)

        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)

        # Generate valid/ambiguous mask from provided mask or depth
        if "mask" not in view_data:
            view_data["mask"] = torch.tensor(depthmap > 0.0, device=view_data["depth"].device)

        non_ambiguous_mask = view_data["mask"].numpy().astype(int)
        non_ambiguous_mask = cv2.resize(
            non_ambiguous_mask,
            (raw_image.shape[1], raw_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        depthmap = np.where(non_ambiguous_mask, depthmap, 0)

        additional = [non_ambiguous_mask]
        image, depthmap, intrinsics, additional = self._crop_resize_if_necessary(
            image=raw_image,
            resolution=resolution,
            depthmap=depthmap,
            intrinsics=intrinsics,
            additional_quantities=additional,
        )
        non_ambiguous_mask = additional[0]

        view = dict(
            img=image,
            depthmap=depthmap,
            camera_pose=c2w_pose,  # cam2world
            camera_intrinsics=intrinsics,
            non_ambiguous_mask=non_ambiguous_mask,
            dataset="A3DScenes",
            label=scene_name,
            instance=os.path.join("images", str(frame_name)),
        )
        return [view]