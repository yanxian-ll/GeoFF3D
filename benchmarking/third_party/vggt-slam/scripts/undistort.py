import cv2
import numpy as np
import os
from glob import glob

K = np.array([
    [458.654, 0, 367.215],
    [0, 457.296, 248.375],
    [0,  0,  1]
], dtype=np.float32)

# Radial-tangential distortion coefficients
dist_coeffs = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05], dtype=np.float32)

datasets = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult", 
           "V1_01_easy", "V1_02_medium", "V1_03_difficult", "V2_01_easy", "V2_02_medium", 
           "V2_03_difficult", "V2_04_difficult", "V2_05_difficult"]

# === Input/output folders ===
for dataset in datasets:
    input_folder = "/home/<user>/Documents/MASt3R-SLAM/datasets/euroc/" + dataset + "/mav0/cam0/data"
    output_folder = "/home/<user>/Documents/MASt3R-SLAM/datasets/euroc/" + dataset + "/mav0/cam0/data_rectified"

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all image paths (adjust extension if needed)
    image_paths = glob(os.path.join(input_folder, '*.png'))

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to read {img_path}")
            continue

        # Undistort
        undistorted = cv2.undistort(img, K, dist_coeffs)

        # Save with same filename in output folder
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, undistorted)

        print(f"Saved: {out_path}")
