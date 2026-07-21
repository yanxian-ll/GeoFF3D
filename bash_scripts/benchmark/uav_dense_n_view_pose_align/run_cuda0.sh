#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DEVICE=0

scripts=(
  mapa.sh
  mapa_ft.sh
  mapa_t_ft.sh
  pi3.sh
  pi3x_ft.sh
  pi3x_p_ft.sh
  pi3x_transup_t.sh
  pi3x_transup_t_yaw.sh
  pi3x_transup_t_woalign.sh
)

for script in "${scripts[@]}"; do
  echo "============================================================"
  echo "[CUDA ${CUDA_DEVICE}] Running ${script}"
  echo "============================================================"
  bash "$SCRIPT_DIR/$script" "$CUDA_DEVICE"
done

echo "[CUDA ${CUDA_DEVICE}] All assigned benchmarks completed."
