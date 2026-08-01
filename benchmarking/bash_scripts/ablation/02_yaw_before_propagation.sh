#!/bin/bash
set -euo pipefail

# Branch A: add the Z-up constrained alignment before prior propagation.
METHOD_NAME="02_yaw_before_propagation"
SPATIAL_PARTITION="footprint_tree"
POST_CHUNK_ALIGN=true
POST_CHUNK_ALIGN_MODE="yaw_translation"
DEPTH_PRIOR="none"
source "$(dirname "$0")/_base.sh"
