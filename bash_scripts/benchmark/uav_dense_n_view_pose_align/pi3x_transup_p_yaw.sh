#!/bin/bash

# Ours-TR with independent scale + Z-yaw + translation alignment for geometry
# and camera poses.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ALIGNMENT_MODE="${ALIGNMENT_MODE:-pose_yaw}"
export RUN_TAG="${RUN_TAG:-geoff3d_p_yaw}"
export BENCHMARK_SCRIPT="benchmarking/dense_n_view/benchmark_absolute_world.py"

exec bash "$SCRIPT_DIR/pi3x_transup_p.sh" "$@"
