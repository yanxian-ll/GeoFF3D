import gtsam
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from gtsam import NonlinearFactorGraph, Values, noiseModel
from gtsam import SL4, PriorFactorSL4, BetweenFactorSL4
from gtsam.symbol_shorthand import X

from vggt_slam.slam_utils import decompose_camera, normalize_to_sl4

def safe_normalize_to_sl4(H, label="homography"):
    H = np.asarray(H, dtype=float)
    if H.shape != (4, 4) or not np.isfinite(H).all():
        print(f"[VGGT-SLAM2][SL4][WARN] invalid {label}; using identity.")
        return np.eye(4)
    try:
        U, S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        print(f"[VGGT-SLAM2][SL4][WARN] SVD failed for {label}; using identity.")
        return np.eye(4)

    max_s = float(np.max(S)) if S.size else 0.0
    min_s = float(np.min(S)) if S.size else 0.0
    if not np.isfinite(max_s) or not np.isfinite(min_s) or max_s <= 0.0 or min_s / max_s < 1e-8:
        print(f"[VGGT-SLAM2][SL4][WARN] singular {label}; singular_values={S}; using identity.")
        return np.eye(4)

    det = float(np.linalg.det(H))
    if not np.isfinite(det) or abs(det) < 1e-12:
        print(f"[VGGT-SLAM2][SL4][WARN] bad determinant for {label}: {det}; using identity.")
        return np.eye(4)
    if det < 0:
        U[:, -1] *= -1.0
        H = U @ np.diag(S) @ Vt
        det = float(np.linalg.det(H))
        if not np.isfinite(det) or det <= 0.0:
            print(f"[VGGT-SLAM2][SL4][WARN] could not make {label} right-handed; using identity.")
            return np.eye(4)
    return H / (det ** 0.25)

class PoseGraph:
    def __init__(self):
        """Initialize a factor graph for Pose3 nodes with BetweenFactors."""
        self.graph = NonlinearFactorGraph()
        self.values = Values()
        inner_noise = 0.05*np.ones(15, dtype=float)
        intra_noise = 0.05*np.ones(15, dtype=float)
        self.inner_submap_noise = noiseModel.Diagonal.Sigmas(inner_noise)
        self.intra_submap_noise = noiseModel.Diagonal.Sigmas(intra_noise)
        self.anchor_noise = noiseModel.Diagonal.Sigmas([1e-6] * 15)
        self.initialized_nodes = set()
        self.num_loop_closures = 0 # Just used for debugging and analysis

        self.auto_cal_H_mats = dict()  # Store homographies estimated by auto-calibration

    def add_homography(self, key, global_h):
        """Add a new homography node to the graph."""
        # print("det(global_h)", np.linalg.det(global_h))
        global_h = safe_normalize_to_sl4(global_h, label=f"node {key}")
        key = X(key)
        if key in self.initialized_nodes:
            print(f"SL4 {key} already exists.")
            return
        self.values.insert(key, SL4(global_h))
        self.initialized_nodes.add(key)

    def add_between_factor(self, key1, key2, relative_h, noise):
        """Add a relative SL4 constraint between two nodes."""
        relative_h = safe_normalize_to_sl4(relative_h, label=f"between {key1}->{key2}")
        key1 = X(key1)
        key2 = X(key2)
        if key1 not in self.initialized_nodes or key2 not in self.initialized_nodes:
            raise ValueError(f"Both poses {key1} and {key2} must exist before adding a factor.")
        self.graph.add(gtsam.BetweenFactorSL4(key1, key2, SL4(relative_h), noise))
    
    def add_prior_factor(self, key, global_h):
        global_h = safe_normalize_to_sl4(global_h, label=f"prior {key}")
        key = X(key)
        if key not in self.initialized_nodes:
            raise ValueError(f"Trying to add prior factor for key {key} but it is not in the graph.")
        self.graph.add(PriorFactorSL4(key, SL4(global_h), self.anchor_noise))

    def get_homography(self, node_id):
        """
        Get the optimized SL4 homography at a specific node.
        :param node_id: The ID of the node.
        :return: gtsam.SL4 homography of the node.
        """

        auto_cal_H = np.eye(4)
        if node_id in self.auto_cal_H_mats:
            auto_cal_H  = self.auto_cal_H_mats[node_id]
        node_id = X(node_id)
        return auto_cal_H @ self.values.atSL4(node_id).matrix()

    def get_projection_matrix(self, node_id):
        """
        Get the optimized SL4 homography at a specific node.
        :param node_id: The ID of the node.
        :return: gtsam.SL4 homography of the node.
        """
        homography = self.get_homography(node_id)
        projection_matrix = np.linalg.inv(homography)
        return projection_matri

    
    def optimize(self, verbose=False):
        """Optimize the graph with Levenberg–Marquardt and print per-factor errors."""
        # Optional verbosity settings
        params = gtsam.LevenbergMarquardtParams()
        if verbose:
            params.setVerbosityLM("SUMMARY")
            params.setVerbosity("ERROR")

        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values, params)

        # --- Initial total error ---
        initial_error = self.graph.error(self.values)
        print(f"Initial total error: {initial_error:.6f}")

        # --- Per-factor initial error ---
        if verbose:
            print("\nInitial per-factor errors:")
            for i in range(self.graph.size()):
                factor = self.graph.at(i)
                try:
                    e = factor.error(self.values)
                    print(f"  Factor {i:3d}: error = {e:.6f}")
                except RuntimeError as ex:
                    print(f"  Factor {i:3d}: error could not be computed ({ex})")

            keys = [gtsam.DefaultKeyFormatter(k) for k in factor.keys()]
            print(f"Factor {i} connects to {keys} with error {e:.6f}")

        # --- Optimize ---
        try:
            result = optimizer.optimize()
        except RuntimeError as exc:
            print(f"[VGGT-SLAM2][SL4][WARN] optimization failed; keeping initial estimates: {exc}")
            return

        # --- Final total error ---
        final_error = self.graph.error(result)
        # print(f"\nFinal total error: {final_error:.6f}")

        # --- Per-factor final error ---
        if verbose:
            print("\nFinal per-factor errors:")
            for i in range(self.graph.size()):
                factor = self.graph.at(i)
                try:
                    e = factor.error(result)
                    print(f"  Factor {i:3d}: error = {e:.6f}")
                except RuntimeError as ex:
                    print(f"  Factor {i:3d}: error could not be computed ({ex})")

        # --- Store optimized values ---
        self.values = result


    def print_estimates(self):
        """Print the optimized poses."""
        for key in sorted(self.initialized_nodes):
            print(f"Homography{key}:\n{self.values.atSL4(key)}\n")
    
    def increment_loop_closure(self):
        """Increment the loop closure count."""
        self.num_loop_closures += 1
    
    def get_num_loops(self):
        """Get the number of loop closures."""
        return self.num_loop_closures

    def update_all_homographies(self, map, auto_cal_H_mats):
        count = 0
        for submap in map.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            for pose_num in range(len(submap.poses)):
                id = int(submap.get_id() + pose_num)
                self.auto_cal_H_mats[id] = np.linalg.inv(auto_cal_H_mats[count])
                count += 1
        assert count == len(auto_cal_H_mats), "Number of auto-calibration homographies does not match number of poses in the map."
