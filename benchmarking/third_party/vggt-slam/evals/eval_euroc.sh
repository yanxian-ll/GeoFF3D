#!/bin/bash

abs_dir="/home/<user>/Documents"
dataset_path="${abs_dir%/}/MASt3R-SLAM/datasets/euroc/"
gt_path="${abs_dir%/}/MASt3R-SLAM/groundtruths/euroc/"

datasets=(
    MH_01_easy
    MH_02_easy
    MH_03_medium
    MH_04_difficult
    MH_05_difficult
    V1_01_easy
    V1_02_medium
    V1_03_difficult
    V2_01_easy
    V2_02_medium
    V2_03_difficult
)

for dataset in ${datasets[@]}; do
    dataset_name="$dataset_path""$dataset"/mav0/cam0/data_rectified
    python main.py --image_folder $dataset_name  --max_loops 1 --conf_threshold 25 --log_results --log_path $dataset.txt --min_disparity 50 --submap_size 16

done


total=0
count=0

for dataset in ${datasets[@]}; do
    dataset_name="${dataset_path}${dataset}/"
    echo "Processing ${dataset_name}"

    # Run evo_ape and extract RMSE translation error
    result=$(evo_ape tum "${gt_path}${dataset}.txt" "${dataset}.txt" -as)
    echo "$result"

    # Extract RMSE value (trans part) using grep/sed
    rmse=$(echo "$result" | grep "rmse" | head -1 | sed -E 's/.*rmse[^0-9]*([0-9.]+).*/\1/')

    if [[ ! -z "$rmse" ]]; then
        total=$(echo "$total + $rmse" | bc -l)
        count=$((count + 1))
    fi
done

if [[ $count -gt 0 ]]; then
    avg=$(echo "$total / $count" | bc -l)
    echo "Average RMSE translation APE over $count runs: $avg"
else
    echo "No valid results to average."
fi

# for dataset in ${datasets[@]}; do
#     dataset_name="$dataset_path""$dataset"/
#     python eval7_scenes_dense.py --dataset "$dataset_path""$dataset" --gt "$gt_path""$dataset".txt --est $dataset.txt --no-viz

# done
