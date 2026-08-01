#!/bin/bash

abs_dir="/home/<user>/Documents"
submap_size=${1:-16} # Default to 16 if not provided
dataset_path="${abs_dir%/}/MASt3R-SLAM/datasets/7-scenes/"
gt_path="${abs_dir%/}/MASt3R-SLAM/groundtruths/7-scenes/"
log_path="$(pwd)/logs/7scenes_results_w${submap_size}.txt"

mkdir -p "$(pwd)/logs"

# Number of full evaluation repetitions
n=5  # Change this to the number of runs you want

datasets=(
    chess
    fire
    heads
    office
    pumpkin
    redkitchen
    stairs
)

# If file doesn't exist, write CSV header
if [ ! -f "$log_path" ]; then
    echo "Run,Dataset,RMSE,RMSE acc,RMSE comp,Chamfer" > "$log_path"
fi

for run in $(seq 1 $n); do
    echo "==== Starting Run $run ===="

    total_rmse=0
    total_rmse_acc=0
    total_rmse_comp=0
    total_chamfer=0
    count=0

    for dataset in "${datasets[@]}"; do
        dataset_name="${dataset_path}${dataset}/seq-01"
        python main.py --image_folder "$dataset_name" --max_loops 1 --min_disparity 50 --conf_threshold 25 --submap_size "$submap_size" --log_results --log_path "$(pwd)/logs/${dataset}_run${run}_w${submap_size}.txt" 
    done

    for dataset in "${datasets[@]}"; do
        echo "Processing $dataset (Run $run)"

        # Run evo_ape and get RMSE
        ape_result=$(evo_ape tum "${gt_path}${dataset}.txt" "$(pwd)/logs/${dataset}_run${run}_w${submap_size}.txt" -as)
        rmse=$(echo "$ape_result" | grep "rmse" | head -1 | sed -E 's/.*rmse[^0-9]*([0-9.]+).*/\1/')

        # Run dense evaluation
        eval_output=$(python evals/eval7_scenes_dense.py --dataset "${dataset_path}${dataset}" --gt "${gt_path}${dataset}.txt" --est "$(pwd)/logs/${dataset}_run${run}_w${submap_size}.txt" --no-viz)

        rmse_acc=$(echo "$eval_output" | grep "RMSE acc" | awk '{print $3}')
        rmse_comp=$(echo "$eval_output" | grep "RMSE comp" | awk '{print $3}')
        chamfer=$(echo "$eval_output" | grep "Chamfer distance" | awk '{print $3}')

        # Handle missing values (in case parsing fails)
        rmse=${rmse:-0}
        rmse_acc=${rmse_acc:-0}
        rmse_comp=${rmse_comp:-0}
        chamfer=${chamfer:-0}

        # Log individual result
        echo "$run,$dataset,$rmse,$rmse_acc,$rmse_comp,$chamfer" >> "$log_path"

        # Accumulate totals
        total_rmse=$(echo "$total_rmse + $rmse" | bc -l)
        total_rmse_acc=$(echo "$total_rmse_acc + $rmse_acc" | bc -l)
        total_rmse_comp=$(echo "$total_rmse_comp + $rmse_comp" | bc -l)
        total_chamfer=$(echo "$total_chamfer + $chamfer" | bc -l)

        count=$((count + 1))
    done

    # Compute averages
    avg_rmse=$(echo "$total_rmse / $count" | bc -l)
    avg_rmse_acc=$(echo "$total_rmse_acc / $count" | bc -l)
    avg_rmse_comp=$(echo "$total_rmse_comp / $count" | bc -l)
    avg_chamfer=$(echo "$total_chamfer / $count" | bc -l)

    # Log averages as a special "Average" row
    echo "$run,Average,$avg_rmse,$avg_rmse_acc,$avg_rmse_comp,$avg_chamfer" >> "$log_path"

    echo "==== Run $run complete ===="
    echo "Average RMSE: $avg_rmse"
    echo "Average RMSE acc: $avg_rmse_acc"
    echo "Average RMSE comp: $avg_rmse_comp"
    echo "Average Chamfer: $avg_chamfer"
done