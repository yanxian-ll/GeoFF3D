import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(description="Process TUM results")
parser.add_argument("--submap_size", type=str, default="32", help="submap size to use")
args = parser.parse_args()

# Path to your 7-scenes results log
log_path = Path.cwd() / ("logs/" "7scenes_results_w" + args.submap_size + ".txt")

# Load the log file
df = pd.read_csv(log_path)

# Remove any average rows (we’ll recalculate)
df = df[df["Dataset"] != "Average"]

# Ensure correct data types
df["Run"] = df["Run"].astype(int)

print("=== Per-Experiment Results (Run x Dataset) ===")
for run in sorted(df["Run"].unique()):
    print(f"\n--- Run {run} ---")
    run_df = df[df["Run"] == run]
    for _, row in run_df.iterrows():
        scene = row["Dataset"]
        rmse = row["RMSE"]
        acc = row["RMSE acc"]
        comp = row["RMSE comp"]
        chamfer = row["Chamfer"]
        print(f"{scene}: RMSE={rmse:.4f}, Acc={acc:.4f}, Comp={comp:.4f}, Chamfer={chamfer:.4f}")

print("\n=== Per-Run Averages ===")
per_run_avg = df.groupby("Run")[["RMSE", "RMSE acc", "RMSE comp", "Chamfer"]].mean()
print(per_run_avg.round(4))

print("\n=== Per-Dataset Averages Across All Runs ===")
per_scene_avg = df.groupby("Dataset")[["RMSE", "RMSE acc", "RMSE comp", "Chamfer"]].mean()
print(per_scene_avg.round(4))

overall_avg = df[["RMSE", "RMSE acc", "RMSE comp", "Chamfer"]].mean()
print("\n=== Overall Averages Across All Runs and Scenes ===")
print(overall_avg.round(4))
