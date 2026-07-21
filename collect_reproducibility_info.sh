#!/usr/bin/env bash

set +e

OUTPUT_DIR="outputs/reproducibility_info"
OUT="$OUTPUT_DIR/reproducibility_system.txt"

mkdir -p "$OUTPUT_DIR"

{
echo "============================================================"
echo "GeoFF3D Reproducibility Environment"
echo "============================================================"
echo "Collection time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo

echo "============================================================"
echo "1. Operating System"
echo "============================================================"
if [ -f /etc/os-release ]; then
    cat /etc/os-release
fi
echo
echo "Kernel:"
uname -a
echo
echo "Architecture:"
uname -m
echo

echo "============================================================"
echo "2. CPU"
echo "============================================================"
if command -v lscpu >/dev/null 2>&1; then
    lscpu
else
    cat /proc/cpuinfo
fi
echo

echo "============================================================"
echo "3. System Memory"
echo "============================================================"
if command -v free >/dev/null 2>&1; then
    free -h
fi
echo
grep -E 'MemTotal|MemAvailable|SwapTotal' /proc/meminfo 2>/dev/null
echo

echo "============================================================"
echo "4. NVIDIA GPU"
echo "============================================================"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi:"
    nvidia-smi
    echo

    echo "GPU summary:"
    nvidia-smi \
        --query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id,compute_cap \
        --format=csv,noheader
else
    echo "nvidia-smi not found."
fi
echo

echo "============================================================"
echo "5. CUDA Toolkit"
echo "============================================================"
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    echo "nvcc not found in PATH."
fi
echo
echo "CUDA_HOME=${CUDA_HOME:-not set}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-not set}"
echo

echo "============================================================"
echo "6. Compilers and Build Tools"
echo "============================================================"
echo "GCC:"
gcc --version 2>/dev/null | head -n 1
echo
echo "G++:"
g++ --version 2>/dev/null | head -n 1
echo
echo "CMake:"
cmake --version 2>/dev/null | head -n 1
echo
echo "Git:"
git --version 2>/dev/null
echo

echo "============================================================"
echo "7. Python Environment"
echo "============================================================"
echo "Python executable:"
which python 2>/dev/null
echo
echo "Python version:"
python --version 2>&1
echo
echo "Pip version:"
python -m pip --version 2>&1
echo
echo "Conda environment:"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-not set}"
echo "CONDA_PREFIX=${CONDA_PREFIX:-not set}"
if command -v conda >/dev/null 2>&1; then
    conda --version
fi
echo

echo "============================================================"
echo "8. PyTorch, CUDA, cuDNN, NCCL"
echo "============================================================"
python - <<'PY'
import os
import platform
import sys

print("Python:", sys.version.replace("\n", " "))
print("Python executable:", sys.executable)
print("Platform:", platform.platform())

try:
    import torch

    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("cuDNN available:", torch.backends.cudnn.is_available())
    print("cuDNN version:", torch.backends.cudnn.version())
    print("cuDNN enabled:", torch.backends.cudnn.enabled)
    print("TF32 matmul enabled:", torch.backends.cuda.matmul.allow_tf32)
    print("TF32 cuDNN enabled:", torch.backends.cudnn.allow_tf32)
    print("Deterministic algorithms:", torch.are_deterministic_algorithms_enabled())

    try:
        print("NCCL version:", torch.cuda.nccl.version())
    except Exception as e:
        print("NCCL version: unavailable:", e)

    print("Visible CUDA devices:", os.environ.get("CUDA_VISIBLE_DEVICES", "not set"))
    print("GPU count visible to PyTorch:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i} name:", props.name)
        print(f"GPU {i} total memory GiB:", props.total_memory / 1024**3)
        print(f"GPU {i} compute capability:", f"{props.major}.{props.minor}")
        print(f"GPU {i} multiprocessors:", props.multi_processor_count)

    print()
    print("PyTorch build configuration:")
    print(torch.__config__.show())

except Exception as e:
    print("Unable to import PyTorch:", repr(e))
PY
echo

echo "============================================================"
echo "9. Relevant Python Libraries"
echo "============================================================"
python - <<'PY'
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

packages = [
    "torch",
    "torchvision",
    "torchaudio",
    "numpy",
    "scipy",
    "opencv-python",
    "opencv-python-headless",
    "pillow",
    "matplotlib",
    "einops",
    "timm",
    "transformers",
    "accelerate",
    "safetensors",
    "huggingface-hub",
    "hydra-core",
    "omegaconf",
    "lightning",
    "pytorch-lightning",
    "open3d",
    "trimesh",
    "pycolmap",
    "rerun-sdk",
    "scikit-learn",
    "pandas",
    "imageio",
]

for package in packages:
    try:
        print(f"{package}: {version(package)}")
    except PackageNotFoundError:
        pass

try:
    import cv2
    print("cv2 runtime version:", cv2.__version__)
except Exception:
    pass
PY
echo

echo "============================================================"
echo "10. Environment Variables Relevant to Training"
echo "============================================================"
env | grep -E \
'^(CUDA|CUDNN|NCCL|OMP|MKL|PYTHON|TORCH|CONDA|HF_|TRANSFORMERS|OPENCV)_' \
| sort
echo

echo "============================================================"
echo "11. Disk Information"
echo "============================================================"
df -h .
echo

echo "============================================================"
echo "12. PyTorch GPU Test"
echo "============================================================"
python - <<'PY'
try:
    import torch

    if torch.cuda.is_available():
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("CUDA tensor test: success")
        print("Result shape:", tuple(y.shape))
        print("Allocated memory MiB:", torch.cuda.memory_allocated() / 1024**2)
    else:
        print("CUDA tensor test: skipped; CUDA unavailable.")
except Exception as e:
    print("CUDA tensor test failed:", repr(e))
PY

} 2>&1 | tee "$OUT"

echo
echo "Saving Python package list..."

python -m pip freeze > "$OUTPUT_DIR/requirements-freeze.txt" 2>&1

if command -v conda >/dev/null 2>&1; then
    conda env export --no-builds > "$OUTPUT_DIR/environment.yml" 2>&1
fi

{
echo "============================================================"
echo "Git Reproducibility Information"
echo "============================================================"
echo "Repository: $(pwd)"
echo
echo "Remote:"
git remote -v 2>/dev/null
echo
echo "Current branch:"
git branch --show-current 2>/dev/null
echo
echo "Commit:"
git rev-parse HEAD 2>/dev/null
echo
echo "Commit description:"
git describe --always --dirty --tags 2>/dev/null
echo
echo "Git status:"
git status --short 2>/dev/null
echo
echo "Submodules:"
git submodule status --recursive 2>/dev/null
} > "$OUTPUT_DIR/git_reproducibility.txt" 2>&1

echo
echo "Generated files:"
echo "  $OUT"
echo "  $OUTPUT_DIR/requirements-freeze.txt"
if command -v conda >/dev/null 2>&1; then
    echo "  $OUTPUT_DIR/environment.yml"
fi
echo "  $OUTPUT_DIR/git_reproducibility.txt"
