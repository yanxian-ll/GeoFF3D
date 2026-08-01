#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
CUDA_DEVICE=0
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then CUDA_DEVICE="$1"; shift; fi
exec python3 "$SCRIPT_DIR/run.py" geoff3d --cuda-device "$CUDA_DEVICE" "$@"
