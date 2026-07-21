


import os

data_dir = "experiments/mapanything/benchmarking"
output_dir = "experiments/mapanything/benchmarking_json_only"
os.makedirs(output_dir, exist_ok=True)

import os
import shutil

for dense_n_view in os.listdir(data_dir):
    if not dense_n_view.startswith("dense_"):
        continue

    for method in os.listdir(os.path.join(data_dir, dense_n_view)):
        method_dir = os.path.join(data_dir, dense_n_view, method)

        for json_file in os.listdir(method_dir):
            if json_file.endswith(".json"):
                src_path = os.path.join(method_dir, json_file)
                out_path = os.path.join(output_dir, dense_n_view, method, json_file)

                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                shutil.copy2(src_path, out_path)
