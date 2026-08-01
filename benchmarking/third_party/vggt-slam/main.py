import os
import sys
import glob
import argparse
import json
import time
from pathlib import Path

# ── Set up local VGGT path BEFORE any vggt_slam imports ──────────
# vggt_slam modules (solver.py, etc.) import vggt at module level.
_VGGT_REPO = os.environ.get("VGGT_REPO")
if not _VGGT_REPO:
    _VGGT_REPO = Path(__file__).resolve().parent / "vggt"
    # Also scan --vggt_repo from raw sys.argv (before argparse parses)
    for i, arg in enumerate(sys.argv):
        if arg == "--vggt_repo" and i + 1 < len(sys.argv):
            _VGGT_REPO = sys.argv[i + 1]
            break
if _VGGT_REPO and Path(_VGGT_REPO).exists() and (Path(_VGGT_REPO) / "vggt").is_dir():
    sys.path.insert(0, str(Path(_VGGT_REPO).resolve()))
    print(f"[VGGT-SLAM] Pre-import local VGGT source: {Path(_VGGT_REPO).resolve()}")
_SALAD_REPO = Path(__file__).resolve().parent / "salad"
if (_SALAD_REPO / "salad").is_dir():
    sys.path.insert(0, str(_SALAD_REPO.resolve()))
    print(f"[VGGT-SLAM] Pre-import local SALAD source: {_SALAD_REPO.resolve()}")
# ──────────────────────────────────────────────────────────────────

import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt_slam.local_weights import configure_vggt_slam_runtime, load_vggt_weights
from vggt_slam.vggt_source import build_vggt_from_local_path


parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")
parser.add_argument("--use_sim3", action="store_true", help="Use Sim3 instead of SL(4)")
parser.add_argument("--plot_focal_lengths", action="store_true", help="Plot focal lengths for the submaps")
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
parser.add_argument("--downsample_factor", type=int, default=1, help="Factor to reduce image size by 1/N")
parser.add_argument("--max_loops", type=int, default=1, help="Maximum number of loop closures per submap")
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--vis_stride", type=int, default=1, help="Stride interval in the 3D point cloud image for visualization. Try increasing (such as 4) to reduce lag in visualizing large maps.")
parser.add_argument("--vis_point_size", type=float, default=0.003, help="Visualization point size")
parser.add_argument(
    "--headless",
    action="store_true",
    help="Disable Viser server and final visualization for batch benchmark/export.",
)
parser.add_argument(
    "--log_global_points",
    action="store_true",
    help="Save one global point cloud instead of framewise dense pointcloud logs.",
)
parser.add_argument(
    "--global_points_path",
    type=str,
    default=None,
    help="Path for global point cloud. Defaults to <log_path>_points.ply.",
)
parser.add_argument(
    "--max_global_points",
    type=int,
    default=500000,
    help="Maximum points in exported global point cloud.",
)
parser.add_argument(
    "--global_voxel_downsample",
    type=float,
    default=0.01,
    help="Voxel size for exported global point cloud; <=0 disables voxel downsampling.",
)
parser.add_argument(
    "--global_point_stride",
    type=int,
    default=1,
    help="Stride on dense point maps before exporting the global point cloud.",
)
parser.add_argument("--vggt_model_path", type=str, default=None, help="Local path to VGGT-1B model.pt")
parser.add_argument("--vggt_repo", type=str, default=None, help="Local VGGT source repo path. Must contain a vggt/ package directory.")
parser.add_argument("--torch_hub_dir", type=str, default=None, help="Local torch.hub directory")
parser.add_argument("--offline", action="store_true", help="Disable online checkpoint downloads")
parser.add_argument("--timing_path", type=str, default=None, help="Optional JSON path for processing time metadata")
parser.add_argument(
    "--disable_keyframe_selection",
    action="store_true",
    help="Use all input frames instead of optical-flow keyframe selection.",
)

def main():
    """
    Main function that wraps the entire pipeline of VGGT-SLAM.
    """
    args = parser.parse_args()
    use_optical_flow_downsample = not args.disable_keyframe_selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Configure torch.hub and load local weights (must happen before Solver init)
    configure_vggt_slam_runtime(
        torch_hub_dir=args.torch_hub_dir,
        offline=args.offline,
    )

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        use_sim3=args.use_sim3,
        gradio_mode=False,
        vis_stride=args.vis_stride,
        vis_point_size=args.vis_point_size,
        headless=args.headless,
    )

    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")

    model = build_vggt_from_local_path(
        vggt_repo=args.vggt_repo,
        default_relative_path="vggt",
    )
    load_vggt_weights(model, model_path=args.vggt_model_path, offline=args.offline)

    model.eval()
    model = model.to(device)

    processing_time_start = time.perf_counter()

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = [f for f in glob.glob(os.path.join(args.image_folder, "*")) 
               if "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower() 
               and "db" not in os.path.basename(f).lower()]

    image_names = utils.sort_images_by_number(image_names)
    image_names = utils.downsample_images(image_names, args.downsample_factor)
    print(f"Found {len(image_names)} images")

    image_names_subset = []
    data = []
    for image_name in tqdm(image_names):
        if use_optical_flow_downsample:
            img = cv2.imread(image_name)
            enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
            if enough_disparity:
                image_names_subset.append(image_name)
        else:
            image_names_subset.append(image_name)

        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(image_names_subset) == args.submap_size + args.overlapping_window_size or image_name == image_names[-1]:
            print(image_names_subset)
            predictions = solver.run_predictions(image_names_subset, model, args.max_loops)

            data.append(predictions["intrinsic"][:,0,0])

            solver.add_points(predictions)

            solver.graph.optimize()
            solver.map.update_submap_homographies(solver.graph)

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()
            
            # Reset for next submap.
            image_names_subset = image_names_subset[-args.overlapping_window_size:]
        
    processing_time_seconds = time.perf_counter() - processing_time_start
    timing_payload = {
        "schema": "processing_time_v1",
        "method": "vggt-slam",
        "processing_time_seconds": float(processing_time_seconds),
        "processing_time_ms": float(processing_time_seconds * 1000.0),
        "start_event": "after_vggt_weights_loaded",
        "end_event": "before_final_visualization_and_result_logging",
        "excluded": ["weight_loading", "final_result_saving"],
        "num_input_images": int(len(image_names)),
        "num_submaps": int(solver.map.get_num_submaps()),
        "num_loop_closures": int(solver.graph.get_num_loops()),
    }
    print(f"Processing time (excluding weights/final saving): {processing_time_seconds:.6f}s")
    if args.timing_path:
        timing_path = Path(args.timing_path).expanduser().resolve()
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(json.dumps(timing_payload, indent=2), encoding="utf-8")
        print(f"Saved processing timing: {timing_path}")

    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())

    if not args.vis_map and not args.headless:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    if args.log_results:
        print(f"Writing poses to {args.log_path}...")
        solver.map.write_poses_to_file(args.log_path)

        if args.log_global_points:
            if args.global_points_path:
                global_points_path = args.global_points_path
            else:
                global_points_path = args.log_path.replace(".txt", "_points.ply")

            print(f"Writing sampled global point cloud to {global_points_path}...")
            solver.map.write_sampled_points_to_file(
                global_points_path,
                max_points=int(args.max_global_points),
                voxel_size=float(args.global_voxel_downsample),
                seed=0,
                point_stride=int(args.global_point_stride),
            )
            print(f"Saved sampled global point cloud: {global_points_path}")

        if not args.skip_dense_log:
            logs_dir = args.log_path.replace(".txt", "_logs")
            print(f"Saving framewise dense pointclouds to {logs_dir}...")
            solver.map.save_framewise_pointclouds(logs_dir)
            print("Saved framewise dense pointclouds.")

    if args.plot_focal_lengths:
        # Define a colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
        # Create the scatter plot
        plt.figure(figsize=(8, 6))
        for i, values in enumerate(data):
            y = values  # Y-values from the list
            x = [i] * len(values)  # X-values (same for all points in the list)
            plt.scatter(x, y, color=colors[i], label=f'List {i+1}')

        plt.xlabel("poses")
        plt.ylabel("Focal lengths")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    main()
