#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
exec python3 "$SCRIPT_DIR/run.py" vggt_slam_sim3 "$@"
