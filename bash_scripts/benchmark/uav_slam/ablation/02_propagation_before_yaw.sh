#!/bin/bash
set -euo pipefail

# Branch B: add prior propagation before the Z-up constrained alignment.
METHOD_NAME="02_propagation_before_yaw"
SPATIAL_PARTITION="footprint_tree"
POST_CHUNK_ALIGN=1
POST_CHUNK_ALIGN_MODE="sim3"
DEPTH_PRIOR="pred"
source "$(dirname "$0")/_base.sh"
