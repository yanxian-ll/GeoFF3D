#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for method in lingbot-map stream3r streamvggt ttt3r; do
  "$SCRIPT_DIR/${method}.sh" "$@"
done
