#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DEVICE=1

scripts=(
  mapa_p.sh
  mapa_p_ft.sh
  pi3_ft.sh
  pi3x_p.sh
  vggt.sh
  vggt_ft.sh
  pi3x_transup_p.sh
  pi3x_transup_p_yaw.sh
  pi3x_transup_p_woalign.sh
)

for script in "${scripts[@]}"; do
  echo "============================================================"
  echo "[CUDA ${CUDA_DEVICE}] Running ${script}"
  echo "============================================================"
  bash "$SCRIPT_DIR/$script" "$CUDA_DEVICE"
done

echo "[CUDA ${CUDA_DEVICE}] All assigned benchmarks completed."
