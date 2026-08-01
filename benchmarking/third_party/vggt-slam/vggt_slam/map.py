import os
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def _project_to_valid_rotation(rotation_matrix):
    R_in = np.asarray(rotation_matrix, dtype=float)[:3, :3]
    if not np.isfinite(R_in).all():
        return np.eye(3)
    try:
        U, _, Vt = np.linalg.svd(R_in)
        R_out = U @ Vt
        if np.linalg.det(R_out) < 0:
            U[:, -1] *= -1
            R_out = U @ Vt
        if not np.isfinite(R_out).all() or np.linalg.det(R_out) <= 0:
            return np.eye(3)
        return R_out
    except np.linalg.LinAlgError:
        return np.eye(3)

class GraphMap:
    def __init__(self):
        self.submaps = dict()
    
    def get_num_submaps(self):
        return len(self.submaps)

    def add_submap(self, submap):
        submap_id = submap.get_id()
        self.submaps[submap_id] = submap
    
    def get_largest_key(self):
        if len(self.submaps) == 0:
            return -1
        return max(self.submaps.keys())
    
    def get_submap(self, id):
        return self.submaps[id]

    def get_latest_submap(self):
        return self.get_submap(self.get_largest_key())
    
    def retrieve_best_score_frame(self, query_vector, current_submap_id, ignore_last_submap=True):
        overall_best_score = 1000
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        # search for best image to target image
        for submap_key in self.submaps.keys():
            if submap_key == current_submap_id:
                continue

            if ignore_last_submap and (submap_key == current_submap_id-1):
                continue

            else:
                submap = self.submaps[submap_key]
                submap_embeddings = submap.get_all_retrieval_vectors()
                scores = []
                for embedding in submap_embeddings:
                    score = torch.linalg.norm(embedding-query_vector)
                    scores.append(score.item())
                
                best_score_id = np.argmin(scores)
                best_score = scores[best_score_id]

                if best_score < overall_best_score:
                    overall_best_score = best_score
                    overall_best_submap_id = submap_key
                    overall_best_frame_index = best_score_id

        return overall_best_score, overall_best_submap_id, overall_best_frame_index

    def get_frames_from_loops(self, loops):
        frames = []
        for detected_loop in loops:
            frames.append(self.submaps[detected_loop.detected_submap_id].get_frame_at_index(detected_loop.detected_submap_frame))
        
        return frames
    
    def update_submap_homographies(self, graph):
        for submap_key in self.submaps.keys():
            submap = self.submaps[submap_key]
            submap.set_reference_homography(graph.get_homography(submap_key).matrix())
    
    def get_submaps(self):
        return self.submaps.values()

    def ordered_submaps_by_key(self):
        for k in sorted(self.submaps):
            yield self.submaps[k]

    def write_poses_to_file(self, file_name):
        with open(file_name, "w") as f:
            for submap in self.ordered_submaps_by_key():
                poses = submap.get_all_poses_world(ignore_loop_closure_frames=True)
                frame_ids = submap.get_frame_ids()
                assert len(poses) == len(frame_ids), "Number of provided poses and number of frame ids do not match"
                for frame_id, pose in zip(frame_ids, poses):
                    x, y, z = pose[0:3, 3]
                    rotation_matrix = pose[0:3, 0:3]
                    rotation_matrix = _project_to_valid_rotation(rotation_matrix)
                    quaternion = R.from_matrix(rotation_matrix).as_quat() # x, y, z, w
                    output = np.array([float(frame_id), x, y, z, *quaternion])
                    f.write(" ".join(f"{v:.8f}" for v in output) + "\n")

    def save_framewise_pointclouds(self, file_name):
        os.makedirs(file_name, exist_ok=True)
        for submap in self.ordered_submaps_by_key():
            pointclouds, frame_ids, conf_masks = submap.get_points_list_in_world_frame(ignore_loop_closure_frames=True)
            for frame_id, pointcloud, conf_masks in zip(frame_ids, pointclouds, conf_masks):
                # save pcd as numpy array
                np.savez(f"{file_name}/{frame_id}.npz", pointcloud=pointcloud, mask=conf_masks)
                

    def _sample_points_and_colors(self, points, colors, max_points=0, seed=0):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)

        n = min(points.shape[0], colors.shape[0])
        points = points[:n]
        colors = colors[:n]

        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        colors = colors[finite]

        if int(max_points) > 0 and points.shape[0] > int(max_points):
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(points.shape[0], size=int(max_points), replace=False)
            points = points[idx]
            colors = colors[idx]

        return points.astype(np.float32), colors.astype(np.uint8)

    def _voxel_downsample_numpy(self, points, colors, voxel_size):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)

        if float(voxel_size) <= 0.0 or points.shape[0] == 0:
            return points, colors

        n = min(points.shape[0], colors.shape[0])
        points = points[:n]
        colors = colors[:n]

        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        colors = colors[finite]

        if points.shape[0] == 0:
            return points, colors

        voxel = np.floor(points.astype(np.float64) / float(voxel_size)).astype(np.int64)
        _, idx = np.unique(voxel, axis=0, return_index=True)

        return points[idx].astype(np.float32), colors[idx].astype(np.uint8)

    def _write_binary_ply(self, file_name, points, colors):
        file_name = str(file_name)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)

        n = min(points.shape[0], colors.shape[0])
        points = points[:n]
        colors = colors[:n]

        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        colors = colors[finite]

        n = int(points.shape[0])
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )

        vertex_dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
        vertices = np.empty(n, dtype=vertex_dtype)
        vertices["x"] = points[:, 0]
        vertices["y"] = points[:, 1]
        vertices["z"] = points[:, 2]
        vertices["red"] = colors[:, 0]
        vertices["green"] = colors[:, 1]
        vertices["blue"] = colors[:, 2]

        with open(file_name, "wb") as f:
            f.write(header.encode("ascii"))
            vertices.tofile(f)

    def write_sampled_points_to_file(
        self,
        file_name,
        max_points=500000,
        voxel_size=0.01,
        seed=0,
        point_stride=1,
        per_submap_factor=2.0,
    ):
        """
        Memory-safe global point cloud export.

        This avoids:
        - concatenating full dense points from all submaps;
        - Open3D double-copying huge arrays;
        - writing a huge PLY that the wrapper later reads back.
        """
        submaps = list(self.ordered_submaps_by_key())
        num_submaps = max(1, len(submaps))

        if int(max_points) > 0:
            per_submap_cap = max(
                1,
                int(np.ceil(float(max_points) * float(per_submap_factor) / num_submaps)),
            )
        else:
            per_submap_cap = 0

        point_parts = []
        color_parts = []

        for submap_i, submap in enumerate(submaps):
            stride = max(1, int(point_stride))

            points = submap.get_points_in_world_frame(stride=stride)
            colors = submap.get_points_colors(stride=stride)

            n_raw = int(points.reshape(-1, 3).shape[0])

            points, colors = self._sample_points_and_colors(
                points,
                colors,
                max_points=per_submap_cap,
                seed=int(seed) + 1009 * (submap_i + 1),
            )

            if float(voxel_size) > 0.0:
                points, colors = self._voxel_downsample_numpy(
                    points,
                    colors,
                    float(voxel_size),
                )

            if points.shape[0] > 0:
                point_parts.append(points)
                color_parts.append(colors)

            print(
                f"[VGGT-SLAM][PLY] submap={submap.get_id()} "
                f"raw={n_raw:,} sampled={points.shape[0]:,} "
                f"stride={stride}"
            )

            # Drop references aggressively before the next submap.
            del points, colors

        if point_parts:
            points = np.concatenate(point_parts, axis=0)
            colors = np.concatenate(color_parts, axis=0)
        else:
            points = np.empty((0, 3), dtype=np.float32)
            colors = np.empty((0, 3), dtype=np.uint8)

        points, colors = self._sample_points_and_colors(
            points,
            colors,
            max_points=int(max_points),
            seed=int(seed),
        )

        if float(voxel_size) > 0.0:
            points, colors = self._voxel_downsample_numpy(
                points,
                colors,
                float(voxel_size),
            )

        self._write_binary_ply(file_name, points, colors)

        print(
            f"[VGGT-SLAM][PLY] saved sampled global point cloud: {file_name}, "
            f"points={points.shape[0]:,}, max_points={int(max_points)}, "
            f"voxel_size={float(voxel_size)}, point_stride={int(point_stride)}"
        )

    def write_points_to_file(self, file_name):
        pcd_all = []
        colors_all = []
        for submap in self.ordered_submaps_by_key():
            pcd = submap.get_points_in_world_frame()
            pcd = pcd.reshape(-1, 3)
            pcd_all.append(pcd)
            colors_all.append(submap.get_points_colors())
        pcd_all = np.concatenate(pcd_all, axis=0)
        colors_all = np.concatenate(colors_all, axis=0)
        if colors_all.max() > 1.0:
            colors_all = colors_all / 255.0
        pcd_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_all))
        pcd_all.colors = o3d.utility.Vector3dVector(colors_all)
        o3d.io.write_point_cloud(file_name, pcd_all)
