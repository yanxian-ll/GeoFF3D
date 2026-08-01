#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for method in vggt_long vggt_slam2.0 vggt_slam_sim3 vggt_slam_sl4; do
  "$SCRIPT_DIR/${method}.sh" "$@"
done
