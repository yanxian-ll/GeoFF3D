import os
import sys
import glob
import time
import argparse
import json
from pathlib import Path

# ── Set up local VGGT path BEFORE any vggt_slam imports ──────────
_VGGT_REPO = os.environ.get("VGGT_REPO")
if not _VGGT_REPO:
    _VGGT_REPO = Path(__file__).resolve().parent / "third_party" / "vggt"
    for i, arg in enumerate(sys.argv):
        if arg == "--vggt_repo" and i + 1 < len(sys.argv):
            _VGGT_REPO = sys.argv[i + 1]
            break
if _VGGT_REPO and Path(_VGGT_REPO).exists() and (Path(_VGGT_REPO) / "vggt").is_dir():
    sys.path.insert(0, str(Path(_VGGT_REPO).resolve()))
    print(f"[VGGT-SLAM2] Pre-import local VGGT source: {Path(_VGGT_REPO).resolve()}")
_SALAD_REPO = Path(__file__).resolve().parent / "third_party" / "salad"
if (_SALAD_REPO / "salad").is_dir():
    sys.path.insert(0, str(_SALAD_REPO.resolve()))
    print(f"[VGGT-SLAM2] Pre-import local SALAD source: {_SALAD_REPO.resolve()}")
# ──────────────────────────────────────────────────────────────────

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt_slam.submap import Submap
from vggt_slam.local_weights import configure_vggt_slam_runtime, load_vggt_weights
from vggt_slam.vggt_source import build_vggt_from_local_path

parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--vis_voxel_size", type=float, default=None, help="Voxel size for downsampling the point cloud in the viewer (e.g. 0.05 for 5 cm). Default: no downsampling")
parser.add_argument("--run_os", action="store_true", help="Enable open-set semantic search with Perception Encoder CLIP and SAM3")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
parser.add_argument("--max_loops", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW or 0 to disable loop closures.")
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--lc_thres", type=float, default=0.95, help="Threshold for image retrieval. Range: [0, 1.0]. Higher = more loop closures")
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
        lc_thres=args.lc_thres,
        vis_voxel_size=args.vis_voxel_size
    )

    print("Initializing and loading VGGT model...")


    if args.run_os:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        sam3_model = build_sam3_image_model()
        processor = Sam3Processor(sam3_model, confidence_threshold=0.50)

        clip_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)  # Downloads from HF
        clip_model = clip_model.cuda()
        clip_tokenizer = transforms.get_text_tokenizer(clip_model.context_length)
        clip_preprocess = transforms.get_image_transform(clip_model.image_size)
    else:
        clip_model, clip_preprocess = None, None
        clip_tokenizer = None

    model = build_vggt_from_local_path(
        vggt_repo=args.vggt_repo,
        default_relative_path="third_party/vggt",
    )
    load_vggt_weights(model, model_path=args.vggt_model_path, offline=args.offline)

    model.eval()
    model = model.to(torch.bfloat16)  # use half precision
    model = model.to(device)

    processing_time_start = time.perf_counter()

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = [f for f in glob.glob(os.path.join(args.image_folder, "*")) 
               if "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower() 
               and "db" not in os.path.basename(f).lower()]

    image_names = utils.sort_images_by_number(image_names)
    downsample_factor = 1
    image_names = utils.downsample_images(image_names, downsample_factor)
    print(f"Found {len(image_names)} images")

    image_names_subset = []
    count = 0
    image_count = 0
    total_time_start = time.time()
    keyframe_time = utils.Accumulator()
    backend_time = utils.Accumulator()
    for image_name in tqdm(image_names):
        if use_optical_flow_downsample:
            with keyframe_time:
                img = cv2.imread(image_name)
                enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
                if enough_disparity:
                    image_names_subset.append(image_name)
                    image_count += 1
        else:
            image_names_subset.append(image_name)
            image_count += 1

        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(image_names_subset) == args.submap_size + args.overlapping_window_size or image_name == image_names[-1]:
            count += 1
            print(image_names_subset)
            t1 = time.time()
            predictions = solver.run_predictions(image_names_subset, model, args.max_loops, clip_model, clip_preprocess)
            print("Solver total time", time.time()-t1)
            print(count, "submaps processed")

            solver.add_points(predictions)

            with backend_time:
                solver.graph.optimize()

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()
            
            # Reset for next submap.
            image_names_subset = image_names_subset[-args.overlapping_window_size:]

    total_time = time.time() - total_time_start
    processing_time_seconds = time.perf_counter() - processing_time_start
    timing_payload = {
        "schema": "processing_time_v1",
        "method": "vggt-slam2.0",
        "processing_time_seconds": float(processing_time_seconds),
        "processing_time_ms": float(processing_time_seconds * 1000.0),
        "start_event": "after_vggt_weights_loaded",
        "end_event": "before_final_visualization_and_result_logging",
        "excluded": ["weight_loading", "final_result_saving"],
        "num_input_images": int(len(image_names)),
        "num_processed_frames": int(image_count),
        "num_submaps": int(solver.map.get_num_submaps()),
        "num_loop_closures": int(solver.graph.get_num_loops()),
        "components": {
            "vggt_seconds": float(solver.vggt_timer.total_time),
            "loop_closure_seconds": float(solver.loop_closure_timer.total_time),
            "keyframe_selection_seconds": float(keyframe_time.total_time),
            "backend_seconds": float(backend_time.total_time),
            "semantic_seconds": float(solver.clip_timer.total_time),
        },
    }
    print(f"Processing time (excluding weights/final saving): {processing_time_seconds:.6f}s")
    if args.timing_path:
        timing_path = Path(args.timing_path).expanduser().resolve()
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(json.dumps(timing_payload, indent=2), encoding="utf-8")
        print(f"Saved processing timing: {timing_path}")

    if image_count > 0:
        average_fps = total_time / image_count
        print(image_count, "frames processed")
        print("Total time:", total_time)
        print(f"Total time for VGGT calls: {solver.vggt_timer.total_time:.4f}s")
        print("Average VGGT time per frame:", solver.vggt_timer.total_time / image_count)
        print("Average loop closure time per frame:", solver.loop_closure_timer.total_time / image_count)
        print("Average keyframe selection time per frame:", keyframe_time.total_time / image_count)
        print("Average backend time per frame:", backend_time.total_time / image_count)
        print("Average semantic time per frame:", solver.clip_timer.total_time / image_count)
        print("Average total time per frame:", total_time / image_count)
        print("Average FPS:", 1 / average_fps)
    else:
        print("No frames processed.")
        
    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())


    if args.run_os:
        while True:
            # Prompt user for text input
            query = input("\nEnter text query or q to quit: ").strip()
            if len(query) == 0:
                print("Empty query. Exiting.")
                return
            
            if query == "q":
                print("Exiting.")
                return
            
            text_emb = utils.compute_text_embeddings(clip_model, clip_tokenizer, query)
            overall_best_score, overall_best_submap_id, overall_best_frame_index = solver.map.retrieve_best_semantic_frame(text_emb)

            found_submap = solver.map.get_submap(overall_best_submap_id)

            # Display image
            best_img = found_submap.get_frame_at_index(overall_best_frame_index)
            print("Score:", overall_best_score)
            with torch.no_grad():
                # convert torch image to PIL
                best_img = to_pil_image(best_img)
                inference_state = processor.set_image(best_img)
                output = processor.set_text_prompt(state=inference_state, prompt=query)
                masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                print(f"Found {masks.shape[0]} masks from SAM3 for the prompt '{query}'")
                print("Scores:", scores.cpu().numpy())


            masked_img = utils.overlay_masks(best_img, masks)
            masked_img.show()

            for i in range(masks.shape[0]):
                mask = masks[i].cpu().numpy()
                obb_center, obb_extent, obb_rotation = utils.compute_obb_from_points(found_submap.get_points_in_mask(overall_best_frame_index, mask, solver.graph))
                solver.viewer.visualize_obb(
                    center=obb_center,
                    extent=obb_extent,
                    rotation=obb_rotation,
                    color=(255, 0, 0),
                    line_width=8.0,
                )

    if not args.vis_map:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    if args.log_results:
        solver.map.write_poses_to_file(args.log_path, solver.graph, kitti_format=False)

        if not args.skip_dense_log:
            # Log the full point cloud as one file
            solver.map.write_points_to_file(solver.graph, args.log_path.replace(".txt", "_points.pcd"))


if __name__ == "__main__":
    main()
