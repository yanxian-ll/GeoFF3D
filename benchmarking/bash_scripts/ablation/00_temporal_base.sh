#!/bin/bash
set -euo pipefail

# Temporal baseline: chronological chunks with 25% adjacent-frame overlap.
# Both per-chunk world anchoring and adjacent chunk-cloud alignment use Sim(3).
METHOD_NAME="00_temporal_base"
SPATIAL_PARTITION="temporal"
CHUNK_ORDER="sequential"
TEMPORAL_OVERLAP_RATIO="${TEMPORAL_OVERLAP_RATIO:-0.25}"
ALIGN="sim3"
POST_CHUNK_ALIGN=true
POST_CHUNK_ALIGN_MODE="sim3"
DEPTH_PRIOR="none"
source "$(dirname "$0")/_base.sh"
