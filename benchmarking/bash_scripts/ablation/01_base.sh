#!/bin/bash
set -euo pipefail

# Base: adaptive footprint tree + hierarchical Sim(3) post-alignment, without
# cross-chunk prior propagation.
METHOD_NAME="01_base"
SPATIAL_PARTITION="footprint_tree"
POST_CHUNK_ALIGN=true
POST_CHUNK_ALIGN_MODE="sim3"
DEPTH_PRIOR="none"
source "$(dirname "$0")/_base.sh"
