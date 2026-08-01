import os
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from vggt_slam.slam_utils import decompose_camera, cosine_similarity

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
        self.rectifying_H_mats = []
        self.non_lc_submap_ids = []
    
    def get_num_submaps(self):
        return len(self.submaps)

    def add_submap(self, submap):
        submap_id = submap.get_id()
        self.submaps[submap_id] = submap
        if not submap.get_lc_status():
            self.non_lc_submap_ids.append(submap_id)
    
    def get_largest_key(self, ignore_loop_closure_submaps=False):
        """
        Get the largest key of the first node of any submap.
        Return: The largest key, or None if the dictionary is empty.
        """
        if len(self.submaps) == 0:
            return None
        if ignore_loop_closure_submaps:
            non_lc_keys = [key for key, submap in self.submaps.items() if not submap.get_lc_status()]
            return max(non_lc_keys)
        return max(self.submaps.keys())
    
    def get_submap(self, id):
        return self.submaps[id]

    def get_latest_submap(self, ignore_loop_closure_submaps=False):
        return self.get_submap(self.get_largest_key(ignore_loop_closure_submaps))

    def retrieve_best_semantic_frame(self, query_text_vector):
        overall_best_score = 0.0
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        # search for best image to target image
        sorted_keys = sorted(self.submaps.keys())
        for index, submap_key in enumerate(sorted_keys):
            submap = self.submaps[submap_key]
            if submap.get_lc_status():
                continue
            submap_embeddings = submap.get_all_semantic_vectors()
            scores = []
            for index, embedding in enumerate(submap_embeddings):
                score = cosine_similarity(embedding, query_text_vector)
                scores.append(score)
            
            best_score_id = np.argmax(scores)
            best_score = scores[best_score_id]

            if best_score > overall_best_score:
                overall_best_score = best_score
                overall_best_submap_id = submap_key
                overall_best_frame_index = best_score_id

        return overall_best_score, overall_best_submap_id, overall_best_frame_index
    
    def retrieve_best_score_frame(self, query_vector, current_submap_id, ignore_last_submap=True):
        overall_best_score = 1000
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        # search for best image to target image
        sorted_keys = sorted(self.submaps.keys())
        for index, submap_key in enumerate(sorted_keys):
            if submap_key == current_submap_id:
                continue

            if self.non_lc_submap_ids and ignore_last_submap and submap_key == self.non_lc_submap_ids[-1]:
                continue

            else:
                submap = self.submaps[submap_key]
                if submap.get_lc_status():
                    continue
                submap_embeddings = submap.get_all_retrieval_vectors()
                scores = []
                for index, embedding in enumerate(submap_embeddings):
                    score = torch.linalg.norm(embedding-query_vector)
                    # score = embedding @ query_vector.t()
                    scores.append(score.item())

                # for now assume we can only have at most one loop closure per submap
                
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
    
    def get_submaps(self):
        return self.submaps.values()

    def ordered_submaps_by_key(self):
        for k in sorted(self.submaps):
            yield self.submaps[k]
    
    def get_all_homographies(self, graph):
        homographies = []
        for submap in self.ordered_submaps_by_key():
            for pose_num in range(len(submap.poses)):
                id = int(submap.get_id() + pose_num)
                homographies.append(graph.get_homography(id))
        return np.stack(homographies)

    def get_all_cam_matricies(self, graph, give_camera_mat):
        cam_mats = []
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            poses = submap.get_all_poses_world(graph, give_camera_mat=give_camera_mat)
            cam_mats.append(poses)
        return np.vstack(cam_mats)

    def write_poses_to_file(self, file_name, graph, give_camera_mat=False, kitti_format=False):
        all_poses = self.get_all_cam_matricies(give_camera_mat=True, graph=graph)
        with open(file_name, "w") as f:

            if self.rectifying_H_mats:
                assert len(self.rectifying_H_mats) == len(all_poses), "Number of rectifying mats and number of poses do not match"
                print("Using rectifying homographies when writing poses to file.")
            count = 0
            for submap_index, submap in enumerate(self.ordered_submaps_by_key()):
                if submap.get_lc_status():
                    continue
                frame_ids = submap.get_frame_ids()
                print(frame_ids)
                for frame_index, frame_id in enumerate(frame_ids):
                    pose = all_poses[count]
                    K, rotation_matrix, t, scale = decompose_camera(pose)
                    # print("Decomposed K:\n", K)
                    count += 1
                    x, y, z = t
                    rotation_matrix = _project_to_valid_rotation(rotation_matrix)
                    if kitti_format:
                        pose_matrix = np.eye(4)
                        pose_matrix[:3, :3] = rotation_matrix
                        pose_matrix[:3, 3] = t
                        output = pose_matrix.flatten()[:-4]
                        output = np.array([float(frame_id), *output])
                    else:
                        quaternion = R.from_matrix(rotation_matrix).as_quat() # x, y, z, w
                        output = np.array([float(frame_id), x, y, z, *quaternion])
                    f.write(" ".join(f"{v:.8f}" for v in output) + "\n")    

    def collect_sampled_points(
        self,
        graph,
        max_points=500000,
        voxel_size=0.01,
        seed=0,
        point_stride=1,
        per_submap_factor=2.0,
        skip_loop_closure_submaps=True,
    ):
        """Collect a sampled global point cloud directly from in-memory submaps.

        This avoids writing a dense PCD and then reading it back in the wrapper.
        """
        def _sample(points, colors, cap, seed_i):
            points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
            colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
            if colors.shape[0] != points.shape[0]:
                colors = np.full((points.shape[0], 3), 220, dtype=np.uint8)

            finite = np.isfinite(points).all(axis=1)
            points = points[finite]
            colors = colors[finite]

            if int(cap) > 0 and points.shape[0] > int(cap):
                rng = np.random.default_rng(int(seed_i))
                idx = rng.choice(points.shape[0], size=int(cap), replace=False)
                points = points[idx]
                colors = colors[idx]
            return points.astype(np.float32), colors.astype(np.uint8)

        def _voxel(points, colors, voxel):
            points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
            colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
            if float(voxel) <= 0 or points.shape[0] == 0:
                return points, colors
            vox = np.floor(points.astype(np.float64) / float(voxel)).astype(np.int64)
            _, idx = np.unique(vox, axis=0, return_index=True)
            return points[idx].astype(np.float32), colors[idx].astype(np.uint8)

        submaps = [
            s for s in self.ordered_submaps_by_key()
            if not (bool(skip_loop_closure_submaps) and s.get_lc_status())
        ]
        num_submaps = max(1, len(submaps))

        if int(max_points) > 0:
            per_submap_cap = max(
                1,
                int(np.ceil(float(max_points) * float(per_submap_factor) / num_submaps)),
            )
        else:
            per_submap_cap = 0

        stride = max(1, int(point_stride))
        pts_parts = []
        col_parts = []

        for submap_i, submap in enumerate(submaps):
            points = submap.get_points_in_world_frame(graph).reshape(-1, 3)
            colors = submap.get_points_colors().reshape(-1, 3)

            if stride > 1:
                points = points[::stride]
                colors = colors[::stride]

            n_raw = int(points.shape[0])
            points, colors = _sample(
                points,
                colors,
                per_submap_cap,
                int(seed) + 1009 * (submap_i + 1),
            )

            if float(voxel_size) > 0:
                points, colors = _voxel(points, colors, float(voxel_size))

            if points.shape[0] > 0:
                pts_parts.append(points)
                col_parts.append(colors)

            print(
                f"[VGGT-SLAM2][POINTS] submap={submap.get_id()} "
                f"raw={n_raw:,} sampled={points.shape[0]:,} stride={stride}"
            )

        if pts_parts:
            points = np.concatenate(pts_parts, axis=0)
            colors = np.concatenate(col_parts, axis=0)
        else:
            points = np.empty((0, 3), dtype=np.float32)
            colors = np.empty((0, 3), dtype=np.uint8)

        points, colors = _sample(points, colors, int(max_points), int(seed))
        if float(voxel_size) > 0:
            points, colors = _voxel(points, colors, float(voxel_size))

        print(
            f"[VGGT-SLAM2][POINTS] collected sampled global point cloud: "
            f"points={points.shape[0]:,}, max_points={int(max_points)}, "
            f"voxel_size={float(voxel_size)}, point_stride={stride}"
        )
        return points.astype(np.float32), colors.astype(np.uint8)

    def write_points_to_file(self, graph, file_name):
        pcd_all = []
        colors_all = []
        for submap in self.ordered_submaps_by_key():
            pcd = submap.get_points_in_world_frame(graph)
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
