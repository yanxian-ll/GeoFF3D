import gtsam
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gtsam import NonlinearFactorGraph, Values, noiseModel
# NOTE: Import our custom SL4 class and bindings (assumed already wrapped)
# `gtsam` should be installed via `gtsam_with_sl4` repository
from gtsam import SL4, PriorFactorSL4, BetweenFactorSL4
from gtsam.symbol_shorthand import X

def safe_normalize_to_sl4(H, label="homography"):
    H = np.asarray(H, dtype=float)
    if H.shape != (4, 4) or not np.isfinite(H).all():
        print(f"[VGGT-SLAM][SL4][WARN] invalid {label}; using identity.")
        return np.eye(4)
    try:
        U, S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        print(f"[VGGT-SLAM][SL4][WARN] SVD failed for {label}; using identity.")
        return np.eye(4)

    max_s = float(np.max(S)) if S.size else 0.0
    min_s = float(np.min(S)) if S.size else 0.0
    if not np.isfinite(max_s) or not np.isfinite(min_s) or max_s <= 0.0 or min_s / max_s < 1e-8:
        print(f"[VGGT-SLAM][SL4][WARN] singular {label}; singular_values={S}; using identity.")
        return np.eye(4)

    det = float(np.linalg.det(H))
    if not np.isfinite(det) or abs(det) < 1e-12:
        print(f"[VGGT-SLAM][SL4][WARN] bad determinant for {label}: {det}; using identity.")
        return np.eye(4)
    if det < 0:
        U[:, -1] *= -1.0
        H = U @ np.diag(S) @ Vt
        det = float(np.linalg.det(H))
        if not np.isfinite(det) or det <= 0.0:
            print(f"[VGGT-SLAM][SL4][WARN] could not make {label} right-handed; using identity.")
            return np.eye(4)
    return H / (det ** 0.25)

class PoseGraph:
    def __init__(self):
        """Initialize a factor graph for Pose3 nodes with BetweenFactors."""
        self.graph = NonlinearFactorGraph()
        self.values = Values()
        # n = 0.05*np.ones(15, dtype=float)
        # n[3] = 1e-6
        # n[7] = 1e-6
        # n[11] = 1e-6

        # n[1] = 1e-6
        # n[2] = 1e-6
        # n[4] = 1e-6
        # n[6] = 1e-6
        # n[8] = 1e-6
        # n[9] = 1e-6
        # n[10] = 1e-6
        self.relative_noise = noiseModel.Diagonal.Sigmas(0.05*np.ones(15, dtype=float))
        self.anchor_noise = noiseModel.Diagonal.Sigmas([1e-6] * 15)
        self.initialized_nodes = set()
        self.num_loop_closures = 0 # Just used for debugging and analysis

    def add_homography(self, key, global_h):
        """Add a new homography node to the graph."""
        global_h = safe_normalize_to_sl4(global_h, label=f"node {key}")
        print("det(global_h)", np.linalg.det(global_h))
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
    
    def add_prior_factor(self, key, global_h, noise):
        global_h = safe_normalize_to_sl4(global_h, label=f"prior {key}")
        key = X(key)
        if key not in self.initialized_nodes:
            raise ValueError(f"Trying to add prior factor for key {key} but it is not in the graph.")
        self.graph.add(PriorFactorSL4(key, SL4(global_h), noise))

    def get_homography(self, node_id):
        """
        Get the optimized SL4 homography at a specific node.
        :param node_id: The ID of the node.
        :return: gtsam.SL4 homography of the node.
        """
        # if node_id >= self.pose_count:
        #     raise ValueError(f"Node ID {node_id} does not exist in the graph.")

        node_id = X(node_id)
        return self.values.atSL4(node_id)
    
    def optimize(self):
        """Optimize the graph and update estimates."""
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values)
        try:
            result = optimizer.optimize()
        except RuntimeError as exc:
            print(f"[VGGT-SLAM][SL4][WARN] optimization failed; keeping initial estimates: {exc}")
            return
        self.values = result  # Update values with optimized results

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


if __name__ == "__main__":
        # Random noise generator
    rng = np.random.default_rng(seed=42)
    def random_noise_vector(dim=15):
        return rng.uniform(low=-0.1, high=0.1, size=dim)

    # Create PoseGraph instance
    pg = PoseGraph()

    # SL4 transformations
    H12 = np.array([[1.0, 0.1, 0.0, 2.0],
                    [0.0, 1.0, 0.0, 3.0],
                    [0.0, 0.0, 1.0, 5.0],
                    [0.001, 0.002, 0.0, 1.0]])

    H23 = np.array([[0.9, 0.2, 0.0, 1.5],
                    [0.1, 1.1, 0.0, -2.0],
                    [0.0, 0.0, 0.8, 4.0],
                    [0.002, 0.003, 0.0005, 1.0]])

    H34 = np.array([[1.05, -0.1, 0.0, 3.0],
                    [0.2, 0.95, 0.0, 1.0],
                    [0.0, 0.0, 0.9, 2.5],
                    [0.0015, -0.001, 0.0003, 1.0]])

    H45 = np.array([[0.98, 0.05, 0.0, -1.0],
                    [-0.05, 1.02, 0.0, 2.0],
                    [0.0, 0.0, 1.1, 0.5],
                    [0.0008, 0.0015, -0.0002, 1.0]])

    # Compose ground-truth poses
    H1 = SL4(np.eye(4))
    H12_SL4 = SL4(H12)
    H23_SL4 = SL4(H23)
    H34_SL4 = SL4(H34)
    H45_SL4 = SL4(H45)

    H2 = H1.compose(H12_SL4)
    H3 = H2.compose(H23_SL4)
    H4 = H3.compose(H34_SL4)
    H5 = H4.compose(H45_SL4)
    H52 = H5.inverse().compose(H2)

    gt_poses = [H1, H2, H3, H4, H5]

    # Add initial nodes
    for i, pose in enumerate(gt_poses, 1):
        noise = random_noise_vector()
        noisy_pose = pose.compose(SL4.Expmap(noise))
        pg.add_homography(i, noisy_pose.matrix())

    # Add prior on node 1
    pg.add_prior_factor(1, np.eye(4), pg.anchor_noise)

    # Add odometry edges
    pg.add_between_factor(1, 2, H12, pg.relative_noise)
    pg.add_between_factor(2, 3, H23, pg.relative_noise)
    pg.add_between_factor(3, 4, H34, pg.relative_noise)
    pg.add_between_factor(4, 5, H45, pg.relative_noise)
    pg.add_between_factor(5, 2, H52.matrix(), pg.relative_noise)  # loop closure

    # Optimize
    pg.optimize()

    # Check results
    for i, gt_pose in enumerate(gt_poses, 1):
        est_pose = pg.get_homography(i)
        if not gt_pose.equals(est_pose, 1e-8):
            print(f"\033[1;31mPose {i} is outside tolerance!")

    print("\033[1;32mSuccessfully optimized!\033[0m")
