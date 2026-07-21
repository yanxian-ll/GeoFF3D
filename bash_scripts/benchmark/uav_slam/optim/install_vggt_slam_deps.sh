#!/bin/bash
set -euo pipefail

# Install dependencies for the optim VGGT family:
#   - VGGT-SLAM 1.0   third_party/vggt-slam
#   - VGGT-SLAM 2.0   third_party/vggt-slam2.0
#   - VGGT-Long       third_party/vggt-long
#
# This script is intended to run inside the existing mapanything conda env.
# By default it does not let pip resolve torch/torchvision dependencies.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP=("$PYTHON_BIN" -m pip)

INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-0}"
INSTALL_OPEN_SET_DEPS="${INSTALL_OPEN_SET_DEPS:-0}"
INSTALL_VGGT_LONG_CPP="${INSTALL_VGGT_LONG_CPP:-0}"
INSTALL_FAISS="${INSTALL_FAISS:-1}"
INSTALL_XFORMERS="${INSTALL_XFORMERS:-0}"
ALLOW_TORCH_DEPS="${ALLOW_TORCH_DEPS:-0}"
OFFLINE="${OFFLINE:-0}"

VGGT_SLAM_DIR="$ROOT_DIR/third_party/vggt-slam"
VGGT_SLAM2_DIR="$ROOT_DIR/third_party/vggt-slam2.0"
VGGT_LONG_DIR="$ROOT_DIR/third_party/vggt-long"

log() {
  printf '\n[VGGT-DEPS] %s\n' "$*"
}

die() {
  echo "[VGGT-DEPS][ERROR] $*" >&2
  exit 1
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "Missing directory: $path"
}

pip_install() {
  "${PIP[@]}" install "$@"
}

pip_install_editable() {
  local path="$1"
  if [[ -d "$path" ]]; then
    if [[ "$ALLOW_TORCH_DEPS" == "1" ]]; then
      pip_install -e "$path"
    else
      pip_install --no-deps -e "$path"
    fi
  else
    echo "[VGGT-DEPS][WARN] skip missing editable package: $path"
  fi
}

install_requirements_filtered() {
  local req_file="$1"
  [[ -f "$req_file" ]] || die "Missing requirements file: $req_file"

  local tmp_req
  tmp_req="$(mktemp)"
  "$PYTHON_BIN" - "$req_file" "$tmp_req" "$INSTALL_XFORMERS" "$INSTALL_FAISS" "$ALLOW_TORCH_DEPS" <<'PY'
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
install_xformers = sys.argv[3] == "1"
install_faiss = sys.argv[4] == "1"
allow_torch_deps = sys.argv[5] == "1"

skip_names = {
    "torch",
    "torchvision",
    "torchaudio",
}
if not allow_torch_deps:
    skip_names.update(
        {
            "lightning",
            "pytorch-lightning",
            "pytorch_metric_learning",
            "pytorch-metric-learning",
            "torchmetrics",
        }
    )
if not install_xformers:
    skip_names.add("xformers")
if not install_faiss:
    skip_names.update({"faiss-gpu", "faiss-cpu", "faiss"})

out = []
for raw in src.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("--extra-index-url"):
        if install_xformers:
            out.append(line)
        continue
    name = re.split(r"[<>=!~\s]", line, maxsplit=1)[0].strip().lower()
    if name in skip_names:
        print(f"[filter] skip {line}")
        continue
    # Avoid forcing exact numpy pins from upstream projects into mapanything.
    if name == "numpy":
        out.append("numpy<2")
        continue
    out.append(line)

dst.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
PY

  if [[ -s "$tmp_req" ]]; then
    pip_install -r "$tmp_req"
  else
    echo "[VGGT-DEPS] no packages left after filtering: $req_file"
  fi
  rm -f "$tmp_req"
}

maybe_clone() {
  local url="$1"
  local dest="$2"
  if [[ -d "$dest" ]]; then
    echo "[VGGT-DEPS] exists, skip clone: $dest"
    return 0
  fi
  if [[ "$OFFLINE" == "1" ]]; then
    echo "[VGGT-DEPS][WARN] offline mode, cannot clone missing repo: $dest"
    return 0
  fi
  git clone --depth 1 "$url" "$dest"
}

install_faiss_fallback() {
  [[ "$INSTALL_FAISS" == "1" ]] || return 0
  log "Installing FAISS dependency"
  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import faiss  # noqa: F401
PY
  then
    echo "[VGGT-DEPS] faiss already importable"
    return 0
  fi

  pip_install faiss-gpu || pip_install faiss-cpu || {
    echo "[VGGT-DEPS][WARN] failed to install faiss-gpu/faiss-cpu; VGGT-Long loop retrieval may fail."
  }
}

install_system_deps() {
  [[ "$INSTALL_SYSTEM_DEPS" == "1" ]] || return 0
  log "Installing system packages"
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "[VGGT-DEPS][WARN] apt-get not found; skip system packages."
    return 0
  fi
  sudo apt-get update
  sudo apt-get install -y git python3-pip libboost-all-dev cmake gcc g++ unzip
}

main() {
  require_dir "$VGGT_SLAM_DIR"
  require_dir "$VGGT_SLAM2_DIR"
  require_dir "$VGGT_LONG_DIR"

  log "Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
  log "Upgrading pip build tools"
  pip_install --upgrade pip "setuptools<81" wheel

  install_system_deps

  log "Installing shared runtime packages"
  pip_install \
    "numpy<2" \
    Pillow \
    huggingface_hub \
    einops \
    safetensors \
    open3d \
    termcolor \
    "viser==0.2.23" \
    tqdm \
    omegaconf \
    opencv-python \
    scipy \
    requests \
    trimesh \
    matplotlib \
    "virtualenv>=20.10.0" \
    lz4 \
    cmake \
    gradio \
    onnxruntime \
    pyparsing \
    importlib_metadata \
    pandas \
    prettytable \
    ftfy \
    regex

  if [[ "$ALLOW_TORCH_DEPS" == "1" ]]; then
    log "Installing torch-adjacent helper packages with dependencies enabled"
    pip_install pytorch_metric_learning pytorch-lightning torchmetrics
  else
    log "Installing torch-adjacent helper packages without dependencies"
    pip_install --no-deps pytorch_metric_learning pytorch-lightning torchmetrics
  fi

  log "Installing upstream requirement files with torch/xformers filtered"
  install_requirements_filtered "$VGGT_SLAM_DIR/requirements.txt"
  install_requirements_filtered "$VGGT_SLAM2_DIR/requirements.txt"
  install_requirements_filtered "$VGGT_LONG_DIR/requirements.txt"
  install_faiss_fallback

  log "Preparing VGGT-SLAM 1.0 local deps"
  maybe_clone https://github.com/Dominic101/salad.git "$VGGT_SLAM_DIR/salad"
  maybe_clone https://github.com/facebookresearch/vggt.git "$VGGT_SLAM_DIR/vggt"
  pip_install_editable "$VGGT_SLAM_DIR/vggt"
  echo "[VGGT-DEPS] skip editable install for VGGT-SLAM 1.0 salad; runner uses local PYTHONPATH."

  log "Preparing VGGT-SLAM 2.0 local deps"
  mkdir -p "$VGGT_SLAM2_DIR/third_party"
  maybe_clone https://github.com/Dominic101/salad.git "$VGGT_SLAM2_DIR/third_party/salad"
  maybe_clone https://github.com/MIT-SPARK/VGGT_SPARK.git "$VGGT_SLAM2_DIR/third_party/vggt"
  pip_install_editable "$VGGT_SLAM2_DIR/third_party/vggt"
  echo "[VGGT-DEPS] skip editable install for VGGT-SLAM 2.0 salad; runner uses local PYTHONPATH."

  if [[ "$INSTALL_OPEN_SET_DEPS" == "1" ]]; then
    log "Installing optional VGGT-SLAM 2.0 open-set dependencies"
    maybe_clone https://github.com/facebookresearch/perception_models.git "$VGGT_SLAM2_DIR/third_party/perception_models"
    maybe_clone https://github.com/facebookresearch/sam3.git "$VGGT_SLAM2_DIR/sam3"
    pip_install_editable "$VGGT_SLAM2_DIR/third_party/perception_models"
    pip_install_editable "$VGGT_SLAM2_DIR/sam3"
  else
    echo "[VGGT-DEPS] skip open-set deps; set INSTALL_OPEN_SET_DEPS=1 to install perception_models/SAM3."
  fi

  log "Preparing VGGT-Long Python deps"
  echo "[VGGT-DEPS] VGGT-Long base_models/vggt is source-only; no editable install needed."
  echo "[VGGT-DEPS] vggt_long.py adds third_party/vggt-long/base_models to sys.path at runtime."

  if [[ "$INSTALL_VGGT_LONG_CPP" == "1" ]]; then
    log "Building optional VGGT-Long C++/CUDA Sim3 solver"
    pip_install ninja
    pip_install -e "$VGGT_LONG_DIR"
  else
    echo "[VGGT-DEPS] skip VGGT-Long C++ solver; pure Python mode is supported."
    echo "[VGGT-DEPS] set INSTALL_VGGT_LONG_CPP=1 to build sim3solve."
  fi

  log "Checking key imports"
  "$PYTHON_BIN" - <<'PY'
import importlib
mods = [
    "torch",
    "cv2",
    "open3d",
    "rerun",
    "viser",
    "omegaconf",
    "vggt",
    "salad",
]
for name in mods:
    try:
        importlib.import_module(name)
        print(f"[OK] import {name}")
    except Exception as exc:
        print(f"[WARN] import {name}: {exc}")
PY
  PYTHONPATH="$VGGT_LONG_DIR/base_models${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
try:
    from base_models.vggt.models.vggt import VGGT  # noqa: F401
    print("[OK] import VGGT-Long source-only base_models.vggt")
except Exception as exc:
    print(f"[WARN] import VGGT-Long source-only base_models.vggt: {exc}")
PY

  cat <<EOF

[VGGT-DEPS] Done.

Notes:
  - Run this inside the mapanything conda env:
      conda activate mapanything
      bash $0
  - This script does not let pip resolve torch/torchvision dependencies by default.
  - Local VGGT weights are expected at:
      $ROOT_DIR/checkpoints/vggt/model.pt
  - Optional knobs:
      INSTALL_SYSTEM_DEPS=1     install apt packages
      INSTALL_OPEN_SET_DEPS=1   install VGGT-SLAM2 open-set deps
      INSTALL_VGGT_LONG_CPP=1   build VGGT-Long C++/CUDA Sim3 solver
      INSTALL_XFORMERS=1        allow xformers installation from requirements
      INSTALL_FAISS=0           skip FAISS installation
      ALLOW_TORCH_DEPS=1        let pip resolve torch-adjacent dependencies
      OFFLINE=1                 avoid git clones
EOF
}

main "$@"
