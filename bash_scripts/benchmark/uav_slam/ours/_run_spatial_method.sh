#!/bin/bash
set -euo pipefail

export HYDRA_FULL_ERROR=1
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-16}"
export OPENCV_IO_ENABLE_OPENEXR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

CLI_CUDA_DEVICE=""
CLI_SCENE_LIST=""
CLI_OVERWRITE=""

print_usage() {
  cat <<EOF
Usage:
  $0 [cuda_device] [scene_list] [--overwrite]
  $0 --cuda-device 0 --scene-list path/to/scenes.yaml --overwrite

Options:
  --overwrite          Re-run scenes even if all expected outputs already exist.
  --no-overwrite       Skip completed scenes. This is the default.
  --cuda-device DEV    CUDA device id, e.g. 0.
  --scene-list PATH    Scene list yaml/path list.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite)
      CLI_OVERWRITE=1
      shift
      ;;
    --no-overwrite)
      CLI_OVERWRITE=0
      shift
      ;;
    --cuda-device|--device)
      CLI_CUDA_DEVICE="${2:?Missing value for $1}"
      shift 2
      ;;
    --cuda-device=*|--device=*)
      CLI_CUDA_DEVICE="${1#*=}"
      shift
      ;;
    --scene-list)
      CLI_SCENE_LIST="${2:?Missing value for $1}"
      shift 2
      ;;
    --scene-list=*)
      CLI_SCENE_LIST="${1#*=}"
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    -*)
      echo "[ERROR] Unknown option: $1" >&2
      print_usage >&2
      exit 2
      ;;
    *)
      # Backward compatible:
      #   script.sh 0 scenes.yaml
      # Also supports:
      #   script.sh scenes.yaml
      if [[ -z "$CLI_CUDA_DEVICE" && "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        CLI_CUDA_DEVICE="$1"
      elif [[ -z "$CLI_SCENE_LIST" ]]; then
        CLI_SCENE_LIST="$1"
      elif [[ -z "$CLI_CUDA_DEVICE" ]]; then
        CLI_CUDA_DEVICE="$1"
      else
        echo "[ERROR] Too many positional arguments: $1" >&2
        print_usage >&2
        exit 2
      fi
      shift
      ;;
  esac
done

if [[ -z "${SCENE_LIST:-}" ]]; then
  cli_scene_list="$CLI_SCENE_LIST"
  method_scene_list="$SCRIPT_DIR/${METHOD_NAME:-}_scenes.yaml"
  if [[ -n "$cli_scene_list" ]]; then
    SCENE_LIST="$cli_scene_list"
  elif [[ -n "${METHOD_NAME:-}" && -f "$method_scene_list" ]]; then
    SCENE_LIST="$method_scene_list"
  else
    SCENE_LIST="$SCRIPT_DIR/../default_scenes.yaml"
  fi
fi
PARAMS_LIST="${PARAMS_LIST:-$SCRIPT_DIR/default_params.yaml}"

python3 - <<'PY'
import sys

try:
    import OpenEXR  # noqa: F401
except Exception as exc:
    print(
        "[ERROR] Missing required Python dependency: OpenEXR\n"
        "        UAV benchmark GT depth uses EXR files. Without OpenEXR, "
        "GT point clouds may be empty and point-cloud metrics will be invalid.\n"
        "        Install with: pip install -e .  or  pip install OpenEXR\n"
        f"        Import error: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(2)
PY

declare -A SCENE_PARAMS=()
declare -A SCENE_DEFAULT_PARAMS=()

param() {
  local key="$1"
  if [[ -v "$key" ]]; then
    printf '%s' "${!key}"
  elif [[ -v "SCENE_PARAMS[$key]" ]]; then
    printf '%s' "${SCENE_PARAMS[$key]}"
  elif [[ -v "SCENE_DEFAULT_PARAMS[$key]" ]]; then
    printf '%s' "${SCENE_DEFAULT_PARAMS[$key]}"
  fi
}

param_default() {
  local key="$1"
  local default_value="$2"
  local value
  value="$(param "$key")"
  if [[ -n "$value" ]]; then
    printf '%s' "$value"
  else
    printf '%s' "$default_value"
  fi
}

param_required() {
  local key="$1"
  local value
  value="$(param "$key")"
  if [[ -z "$value" ]]; then
    echo "[ERROR] Missing required parameter: $key" >&2
    echo "        Set it in $SCENE_LIST params or export it before sourcing this script." >&2
    exit 2
  fi
  printf '%s' "$value"
}

is_on() {
  local value="${1:-0}"
  case "${value,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

cmd_arg() {
  local key="$1"
  local flag="$2"
  local value
  value="$(param "$key")"
  if [[ -n "$value" ]]; then
    CMD+=("$flag" "$value")
  fi
}

SPATIAL_RESULT_MISSING=()

require_result_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    SPATIAL_RESULT_MISSING+=("$path")
  fi
}

spatial_scene_result_complete() {
  local output_rrd="$1"
  local result_dir="${output_rrd%.*}"
  local eval_dir="$result_dir/eval"

  SPATIAL_RESULT_MISSING=()

  # Core outputs from predict_scene_to_rrd_spatial.py / save_spatial_rrd().
  require_result_file "$output_rrd"
  require_result_file "${output_rrd%.*}.json"
  require_result_file "$result_dir/processing_time.json"

  require_result_file "$eval_dir/meta.json"
  require_result_file "$eval_dir/pred_cameras.npz"
  require_result_file "$eval_dir/gt_cameras.npz"
  require_result_file "$eval_dir/pred_points.ply"
  require_result_file "$eval_dir/gt_points.ply"

  # Metrics are part of the benchmark result when RUN_METRICS=1.
  if is_on "$(param_default RUN_METRICS 0)"; then
    require_result_file "$eval_dir/metrics.json"
    require_result_file "$eval_dir/metrics_summary.csv"
  fi

  if is_on "$(param COMPUTE_SEAM_ERROR)"; then
    local sidecar="${output_rrd%.*}.json"
    if [[ -s "$sidecar" ]]; then
      if ! python3 - "$sidecar" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
grid = data.get("grid", {})
seam = grid.get("seam_error", {})
valid = (
    "seam_error" in seam
    and "seam_error_z" in seam
    and int(seam.get("num_adjacency_edges", 0)) > 0
)
if grid.get("partition") == "footprint_grid":
    valid = (
        valid
        and grid.get("alignment_topology") == "parent_graph"
        and float(grid.get("core_coverage_ratio", 0.0)) >= 0.999999
    )
raise SystemExit(0 if valid else 1)
PY
      then
        SPATIAL_RESULT_MISSING+=("$sidecar:grid.seam_error")
      fi
    fi
  fi

  # A run with chunk logging enabled is complete only when every PLY listed in
  # the sidecar still exists. This also makes old runs without chunk artifacts
  # rerun automatically instead of being incorrectly skipped.
  if is_on "$(param LOG_CHUNKS)"; then
    local sidecar="${output_rrd%.*}.json"
    if [[ -s "$sidecar" ]]; then
      if ! python3 - "$sidecar" <<'PY'
import json
import sys
from pathlib import Path

data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
artifacts = data.get("chunk_point_artifacts", [])
valid = bool(artifacts) and all(
    Path(str(item.get("ply_path", ""))).is_file()
    and Path(str(item.get("ply_path", ""))).stat().st_size > 0
    for item in artifacts
)
raise SystemExit(0 if valid else 1)
PY
      then
        SPATIAL_RESULT_MISSING+=("$sidecar:chunk_point_artifacts")
      fi
    fi
  fi

  if is_on "$(param KEEP_CHUNK_CACHE)"; then
    local cache_dir="${output_rrd%.*}/chunk_cache"
    require_result_file "$cache_dir/manifest.json"
    if [[ -s "$cache_dir/manifest.json" ]]; then
      if ! python3 - "$cache_dir/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
records = json.loads(manifest_path.read_text(encoding="utf-8"))
required = {
    "indices", "overlap_indices", "adjacent_chunk_ids",
    "post_chunk_align_transform",
}
valid = bool(records)
for record in records:
    cache_path = Path(str(record.get("chunk_cache_path", "")))
    valid = valid and required.issubset(record) and cache_path.is_file()
    valid = valid and cache_path.stat().st_size > 0 if cache_path.is_file() else False
raise SystemExit(0 if valid else 1)
PY
      then
        SPATIAL_RESULT_MISSING+=("$cache_dir:reproducible_chunk_cache")
      fi
    fi
  fi

  # Optional 3DGS refinement result.
  if is_on "$(param GSPLAT_REFINE)"; then
    require_result_file "$result_dir/gsplat/summary.json"
  fi

  if is_on "$(param EXPORT_TSDF_MESH)"; then
    local mesh_dir
    mesh_dir="$(param TSDF_OUTPUT_DIR)"
    if [[ -z "$mesh_dir" ]]; then
      mesh_dir="$result_dir/mesh"
    fi
    require_result_file "$mesh_dir/tsdf_mesh.ply"
    require_result_file "$mesh_dir/tsdf_mesh_post.ply"
    require_result_file "$mesh_dir/summary.json"
  fi

  if is_on "$(param BUNDLE_ADJUSTMENT)"; then
    local ba_dir
    ba_dir="$(param BA_OUTPUT_DIR)"
    if [[ -z "$ba_dir" ]]; then
      ba_dir="$result_dir/bundle_adjustment"
    fi
    require_result_file "$ba_dir/summary.json"
    require_result_file "$ba_dir/sparse/cameras.bin"
    require_result_file "$ba_dir/sparse/images.bin"
    require_result_file "$ba_dir/sparse/points3D.bin"
  fi

  # Optional DOM result.
  # _run_spatial_method.sh currently writes DOM to output_rrd.with_suffix("")/orthodom.
  if is_on "$(param RENDER_DOM)"; then
    local dom_dir="$result_dir/orthodom"
    require_result_file "$dom_dir/meta.json"
    require_result_file "$dom_dir/dom_rgb.png"
    require_result_file "$dom_dir/dom_alpha.png"
    require_result_file "$dom_dir/dom_dsm.npy"
  fi

  [[ "${#SPATIAL_RESULT_MISSING[@]}" -eq 0 ]]
}

emit_scene_entries() {
  python3 - "$SCENE_LIST" "$PARAMS_LIST" <<'PY'
import base64
import json
import os
import re
import signal
import sys
from pathlib import Path

signal.signal(signal.SIGPIPE, signal.SIG_DFL)


scene_list = Path(sys.argv[1]).expanduser()
params_paths = [
    Path(item).expanduser()
    for arg in sys.argv[2:]
    for item in str(arg).split(":")
    if str(item).strip()
]


def strip_inline_comment(s: str) -> str:
    in_single = False
    in_double = False
    for i, ch in enumerate(s):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if i == 0 or s[i - 1].isspace():
                return s[:i].rstrip()
    return s.rstrip()


def read_config_lines(path: Path):
    out = []
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        line = strip_inline_comment(raw.rstrip("\n"))
        if line.strip():
            out.append(line)
    return out


scene_lines = read_config_lines(scene_list)
raw_lines = []
for params_path in params_paths:
    raw_lines.extend(read_config_lines(params_path))
raw_lines.extend(scene_lines)


def emit(entry):
    payload = json.dumps(entry, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    print(base64.b64encode(payload).decode("ascii"))


def norm_key(key: object) -> str:
    out = re.sub(r"[^0-9A-Za-z]+", "_", str(key).strip()).strip("_").upper()
    return out


def scalar_to_str(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def parse_scalar(value: str):
    value = value.strip()
    if value == "":
        return ""
    if (value[0:1], value[-1:]) in {('"', '"'), ("'", "'")}:
        return value[1:-1]
    low = value.lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    if low in {"null", "none", "~"}:
        return None
    if re.fullmatch(r"[-+]?\d+", value):
        try:
            return int(value)
        except Exception:
            pass
    if (
        re.fullmatch(r"[-+]?(\d+\.\d*|\.\d+)([eE][-+]?\d+)?", value)
        or re.fullmatch(r"[-+]?\d+[eE][-+]?\d+", value)
    ):
        try:
            return float(value)
        except Exception:
            pass
    return value


def split_key_value(text: str):
    if ":" not in text:
        return text.strip(), None
    key, value = text.split(":", 1)
    return key.strip(), value.strip()


lines = [(len(line) - len(line.lstrip(" ")), line.lstrip(" ")) for line in raw_lines]


def parse_block(i: int, indent: int):
    if i >= len(lines):
        return None, i
    if lines[i][0] < indent:
        return None, i
    is_list = lines[i][0] == indent and lines[i][1].startswith("- ")
    if is_list:
        out = []
        while i < len(lines):
            cur_indent, text = lines[i]
            if cur_indent < indent:
                break
            if cur_indent != indent or not text.startswith("- "):
                break
            item_text = text[2:].strip()
            i += 1
            if not item_text:
                child, i = parse_block(i, indent + 2)
                out.append(child)
                continue
            key, value = split_key_value(item_text)
            if value is not None:
                item = {key: parse_scalar(value)} if value != "" else {key: None}
                if value == "":
                    child, i = parse_block(i, indent + 2)
                    item[key] = child
                if i < len(lines) and lines[i][0] > indent:
                    child, i = parse_block(i, indent + 2)
                    if isinstance(child, dict):
                        item.update(child)
                    elif child is not None:
                        item.setdefault("items", child)
                out.append(item)
            else:
                out.append(parse_scalar(item_text))
        return out, i

    out = {}
    while i < len(lines):
        cur_indent, text = lines[i]
        if cur_indent < indent:
            break
        if cur_indent != indent or text.startswith("- "):
            break
        key, value = split_key_value(text)
        i += 1
        if value is None:
            out[key] = None
        elif value == "":
            child, i = parse_block(i, indent + 2)
            out[key] = child
        else:
            out[key] = parse_scalar(value)
    return out, i


def collect_params(obj, out):
    if not isinstance(obj, dict):
        return
    for key, value in obj.items():
        if isinstance(value, dict):
            collect_params(value, out)
        elif value is not None:
            out[norm_key(key)] = scalar_to_str(value)


def params_from(*items):
    params = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for field in ("params", "overrides", "defaults", "variables"):
            values = item.get(field, None)
            if isinstance(values, dict):
                collect_params(values, params)
    return params


def enabled(item) -> bool:
    if not isinstance(item, dict):
        return True
    value = item.get("enabled", True)
    if isinstance(value, str):
        return value.lower() not in {"0", "false", "no", "off"}
    return bool(value)


def join_scene_path(root: str, path: str) -> str:
    path = os.path.expanduser(str(path))
    if os.path.isabs(path) or not root:
        return path
    return os.path.join(os.path.expanduser(str(root)), path)


if not scene_lines:
    sys.exit(0)

first = scene_lines[0].strip()
if first.startswith(("/", "~", ".")):
    for scene_dir in scene_lines:
        scene_dir = scene_dir.strip()
        emit(
            {
                "dataset": "default",
                "scene": Path(scene_dir).name,
                "scene_dir": scene_dir,
                "params": {},
            }
        )
    sys.exit(0)

root_obj, next_i = parse_block(0, lines[0][0])
if next_i < len(lines):
    raise SystemExit(f"Could not parse scene list near line: {raw_lines[next_i]!r}")

if isinstance(root_obj, list):
    for item in root_obj:
        if isinstance(item, str):
            emit(
                {
                    "dataset": "default",
                    "scene": Path(item).name,
                    "scene_dir": item,
                    "params": {},
                }
            )
    sys.exit(0)

if not isinstance(root_obj, dict):
    raise SystemExit("Scene list must be a path list or a mapping with datasets.")

top_params = params_from(root_obj)
datasets = root_obj.get("datasets", [])
if isinstance(datasets, dict):
    datasets = [
        {"name": name, **(cfg if isinstance(cfg, dict) else {"root": cfg})}
        for name, cfg in datasets.items()
    ]

for dataset in datasets:
    if isinstance(dataset, str):
        dataset = {"name": Path(dataset).name, "root": dataset, "scenes": []}
    if not isinstance(dataset, dict) or not enabled(dataset):
        continue
    root = str(dataset.get("root", dataset.get("path", dataset.get("dir", ""))) or "")
    dataset_name = str(dataset.get("name", "") or (Path(root).name if root else "dataset"))
    dataset_params = params_from(dataset)
    scenes = dataset.get("scenes", [])
    if isinstance(scenes, dict):
        scenes = [
            {"name": name, **(cfg if isinstance(cfg, dict) else {"path": cfg})}
            for name, cfg in scenes.items()
        ]
    for scene in scenes:
        if isinstance(scene, str):
            scene_path = join_scene_path(root, scene)
            scene_name = Path(scene_path).name
            scene_params = {}
        elif isinstance(scene, dict):
            if not enabled(scene):
                continue
            raw_path = scene.get("scene_dir", scene.get("path", scene.get("dir", None)))
            scene_name = str(scene.get("name", "") or "")
            if raw_path is None:
                if not scene_name:
                    raise SystemExit(f"Scene in dataset {dataset_name!r} is missing name/path.")
                raw_path = scene_name
            scene_path = join_scene_path(root, str(raw_path))
            if not scene_name:
                scene_name = Path(scene_path).name
            scene_params = params_from(scene)
        else:
            continue
        params = {}
        params.update(dataset_params)
        params.update(scene_params)
        emit(
            {
                "dataset": dataset_name,
                "scene": scene_name,
                "scene_dir": scene_path,
                "defaults": top_params,
                "params": params,
            }
        )
PY
}

load_scene_entry() {
  local entry_b64="$1"
  eval "$(
    python3 - "$entry_b64" <<'PY'
import base64
import json
import shlex
import sys


entry = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))


def q(value):
    return shlex.quote(str(value))


print(f"DATASET_NAME={q(entry.get('dataset', 'default'))}")
print(f"SCENE_NAME={q(entry.get('scene', 'scene'))}")
print(f"SCENE_DIR={q(entry.get('scene_dir', ''))}")
print("declare -gA SCENE_DEFAULT_PARAMS=()")
print("declare -gA SCENE_PARAMS=()")
for key, value in sorted((entry.get("defaults") or {}).items()):
    print(f"SCENE_DEFAULT_PARAMS[{key}]={q(value)}")
for key, value in sorted((entry.get("params") or {}).items()):
    print(f"SCENE_PARAMS[{key}]={q(value)}")
PY
  )"
}

evaluate_spatial_result() {
  local scene_dir="$1"
  local output_rrd="$2"
  local eval_dir="${3:-${output_rrd%.*}/eval}"

  if ! is_on "$(param_default RUN_METRICS 0)"; then
    echo "[METRICS] Skip metrics because RUN_METRICS=$(param_default RUN_METRICS 0)"
    return 0
  fi

  if [[ ! -f "${eval_dir}/pred_cameras.npz" ]]; then
    echo "[METRICS][WARN] Missing ${eval_dir}/pred_cameras.npz, skip."
    return 0
  fi

  if [[ ! -f "${eval_dir}/pred_points.ply" ]]; then
    echo "[METRICS][WARN] Missing ${eval_dir}/pred_points.ply, skip."
    return 0
  fi

  local metric_gt_io_workers
  metric_gt_io_workers="$(param_default METRIC_GT_IO_WORKERS "$(param SCENE_IO_WORKERS)")"

  python "$ROOT_DIR/scripts/evaluate_aligned_reconstruction_metrics.py" \
    --scene_dir "${scene_dir}" \
    --eval_dir "${eval_dir}" \
    --output_json "${eval_dir}/metrics.json" \
    --output_csv "${eval_dir}/metrics_summary.csv" \
    --max_side "$(param MAX_SIDE)" \
    --thresholds "$(param METRIC_THRESHOLDS)" \
    --rpe_steps "$(param METRIC_RPE_STEPS)" \
    --max_gt_points "$(param METRIC_MAX_GT_POINTS)" \
    --max_pred_points_eval "$(param METRIC_MAX_PRED_POINTS_EVAL)" \
    --max_gt_points_eval "$(param METRIC_MAX_GT_POINTS_EVAL)" \
    --max_align_points "$(param METRIC_MAX_ALIGN_POINTS)" \
    --icp_iterations "$(param METRIC_ICP_ITERATIONS)" \
    --icp_trim_quantile "$(param METRIC_ICP_TRIM_QUANTILE)" \
    --gt_io_workers "${metric_gt_io_workers}" \
    --seed 0
}

scene_count=0
while IFS= read -r scene_entry_b64; do
  [[ -z "$scene_entry_b64" ]] && continue
  load_scene_entry "$scene_entry_b64"
  scene_count=$((scene_count + 1))

  METHOD_NAME_RUN="$(param_required METHOD_NAME)"
  MODEL_RUN="$(param_required MODEL)"
  OUTPUT_BASE_RUN="$(param_default OUTPUT_BASE "$ROOT_DIR/outputs/spatial/${METHOD_NAME_RUN}")"
  CHECKPOINT_RUN="$(param_required CHECKPOINT)"
  if [[ -n "$CHECKPOINT_RUN" && "$CHECKPOINT_RUN" != /* ]]; then
    CHECKPOINT_RUN="$ROOT_DIR/$CHECKPOINT_RUN"
  fi
  if [[ -n "$CHECKPOINT_RUN" && ! -f "$CHECKPOINT_RUN" ]]; then
    echo "[ERROR] Checkpoint file not found: $CHECKPOINT_RUN" >&2
    echo "        Set CHECKPOINT to the fine-tuned checkpoint file before running:" >&2
    echo "        CHECKPOINT=/absolute/path/to/checkpoint-best.pth bash $0 ..." >&2
    exit 2
  fi
  if [[ -n "$CLI_CUDA_DEVICE" ]]; then
    CUDA_DEVICE_RUN="$CLI_CUDA_DEVICE"
  else
    CUDA_DEVICE_RUN="$(param_default CUDA_DEVICE 0)"
  fi
  OUTPUT_GROUP_BY_DATASET_RUN="$(param_default OUTPUT_GROUP_BY_DATASET 1)"

  if [[ ! -d "$SCENE_DIR" ]]; then
    echo "[WARN] Scene directory not found, skipping: $SCENE_DIR"
    continue
  fi

  scene_name="$SCENE_NAME"
  [[ -z "$scene_name" ]] && scene_name="$(basename "$SCENE_DIR")"

  if [[ "$DATASET_NAME" != "default" ]] && is_on "$OUTPUT_GROUP_BY_DATASET_RUN"; then
    output_dir="$OUTPUT_BASE_RUN/$DATASET_NAME"
  else
    output_dir="$OUTPUT_BASE_RUN"
  fi
  mkdir -p "$output_dir"
  output_rrd="$output_dir/${scene_name}.rrd"

  OVERWRITE_RUN="${CLI_OVERWRITE:-$(param_default OVERWRITE 0)}"
  if ! is_on "$OVERWRITE_RUN"; then
    if spatial_scene_result_complete "$output_rrd"; then
      echo ""
      echo ">>> Skipping completed scene: $DATASET_NAME/$scene_name"
      echo "    Output RRD: $output_rrd"
      echo "    Reason:     all expected result files exist. Use --overwrite or OVERWRITE=1 to rerun."
      echo "<<< Skipped scene: $scene_name"
      continue
    elif [[ -e "$output_rrd" || -d "${output_rrd%.*}" ]]; then
      echo ""
      echo ">>> Incomplete existing result, rerun scene: $DATASET_NAME/$scene_name"
      echo "    Output RRD: $output_rrd"
      echo "    Missing:"
      for missing_path in "${SPATIAL_RESULT_MISSING[@]}"; do
        echo "      - $missing_path"
      done
    fi
  else
    echo ""
    echo ">>> Overwrite enabled, rerun scene: $DATASET_NAME/$scene_name"
    echo "    Output RRD: $output_rrd"
  fi

  echo ""
  echo ">>> Processing scene: $DATASET_NAME/$scene_name"
  echo "    Method:     $METHOD_NAME_RUN"
  echo "    Model:      $MODEL_RUN"
  echo "    Scene dir:  $SCENE_DIR"
  echo "    Output RRD: $output_rrd"
  echo "    Params:     stride=$(param STRIDE) max_chunk=$(param MAX_CHUNK_SIZE) min_chunk=$(param MIN_CHUNK_SIZE) order=$(param CHUNK_ORDER) norm=$(param NORM_TYPE) patch=$(param PATCH_SIZE)"

  CMD=(
    python3 "$ROOT_DIR/scripts/predict_scene_to_rrd_spatial.py"
    --model "$MODEL_RUN"
    --scene_dir "$SCENE_DIR"
    --output_rrd "$output_rrd"
    --device "cuda:$CUDA_DEVICE_RUN"
    --seed 0
  )

  cmd_arg NUM_VIEWS --num_views
  cmd_arg STRIDE --stride
  cmd_arg IMAGES_DIR --images_dir
  cmd_arg CAMS_DIR --cams_dir
  cmd_arg DEPTH_DIR --depth_dir
  cmd_arg DEPTH_SCALE --depth_scale
  cmd_arg DEPTH_MIN --depth_min
  cmd_arg DEPTH_MAX --depth_max
  cmd_arg MAX_SIDE --max_side
  cmd_arg NORM_TYPE --norm_type
  cmd_arg PATCH_SIZE --patch_size
  cmd_arg SCENE_IO_WORKERS --scene_io_workers
  cmd_arg FOOTPRINT_WORKERS --footprint_workers
  cmd_arg CHUNK_CACHE_WORKERS --chunk_cache_workers
  cmd_arg CHUNK_CACHE_MAX_PENDING --chunk_cache_max_pending
  cmd_arg CHUNK_ORDER --chunk_order
  cmd_arg SPATIAL_PARTITION --spatial_partition
  cmd_arg FOOTPRINT_ESTIMATION --footprint_estimation
  cmd_arg POSE_GRID_SIZE --pose_grid_size
  cmd_arg POSE_GRID_NEIGHBOR_RADIUS --pose_grid_neighbor_radius
  cmd_arg TEMPORAL_OVERLAP_RATIO --temporal_overlap_ratio
  cmd_arg PREDICTED_FOOTPRINT_SAMPLE_STRIDE --predicted_footprint_sample_stride
  cmd_arg PREDICTED_FOOTPRINT_MIN_POINTS --predicted_footprint_min_points
  cmd_arg PREDICTED_FOOTPRINT_QUANTILE_MIN --predicted_footprint_quantile_min
  cmd_arg PREDICTED_FOOTPRINT_QUANTILE_MAX --predicted_footprint_quantile_max
  cmd_arg CHUNK_FOOTPRINT_POINT_SIZE --chunk_footprint_point_size
  cmd_arg CHUNK_FOOTPRINT_BG_POINT_SIZE --chunk_footprint_bg_point_size
  cmd_arg CHUNK_FOOTPRINT_ALPHA --chunk_footprint_alpha
  cmd_arg CHUNK_FOOTPRINT_BG_ALPHA --chunk_footprint_bg_alpha
  cmd_arg CHUNK_FOOTPRINT_LABEL_SIZE --chunk_footprint_label_size
  cmd_arg CHUNK_FOOTPRINT_FONT_SCALE --chunk_footprint_font_scale
  cmd_arg CHUNK_FOOTPRINT_PADDING_RATIO --chunk_footprint_padding_ratio
  if is_on "$(param CHUNK_FOOTPRINT_SHOW_LEGEND)"; then
    CMD+=(--chunk_footprint_show_legend)
  fi
  cmd_arg CHUNK_FOOTPRINT_LEGEND_COLS --chunk_footprint_legend_cols
  cmd_arg CHUNK_FOOTPRINT_LEGEND_MAX_ROWS --chunk_footprint_legend_max_rows
  cmd_arg MAX_CHUNK_SIZE --max_chunk_size
  cmd_arg MIN_CHUNK_SIZE --min_chunk_size
  cmd_arg MODEL_FAMILY --model_family
  cmd_arg ALIGN --align
  cmd_arg RECENTER --recenter
  cmd_arg POSE_PRIOR --pose_prior
  cmd_arg TRANSLATION_PRIOR --translation_prior
  cmd_arg ROTATION_PRIOR --rotation_prior
  cmd_arg RAY_PRIOR --ray_prior
  cmd_arg DEPTH_PRIOR --depth_prior
  cmd_arg MAX_POINTS_PER_VIEW --max_points_per_view
  cmd_arg VOXEL_DOWNSAMPLE --voxel_downsample
  cmd_arg CONF_QUANTILE --conf_quantile
  if ! is_on "$(param_default DEPTH_CONF_FILTER 1)"; then
    CMD+=(--no_depth_conf_filter)
  fi
  if is_on "$(param DEBUG_DEPTH_CONF_FILTER)"; then
    CMD+=(--debug_depth_conf_filter)
  fi
  if is_on "$(param XY_FILL_UNMASKED)"; then
    CMD+=(--xy_fill_unmasked)
  fi
  cmd_arg XY_FILL_GRID_SIZE --xy_fill_grid_size
  cmd_arg XY_FILL_MAX_POINTS_PER_CHUNK --xy_fill_max_points_per_chunk
  if ! is_on "$(param_default POINT_DOWNSAMPLE 1)"; then
    CMD+=(--no_point_downsample)
  fi

  if is_on "$(param POSE_PERTURB)"; then
    CMD+=(--pose_perturb)
    cmd_arg POSE_PERTURB_XY_STD --pose_perturb_xy_std
    cmd_arg POSE_PERTURB_Z_STD --pose_perturb_z_std
    cmd_arg POSE_PERTURB_YAW_STD_DEG --pose_perturb_yaw_std_deg
    cmd_arg POSE_PERTURB_XY_MAX --pose_perturb_xy_max
    cmd_arg POSE_PERTURB_Z_MAX --pose_perturb_z_max
    cmd_arg POSE_PERTURB_YAW_MAX_DEG --pose_perturb_yaw_max_deg
    cmd_arg POSE_PERTURB_SEED_OFFSET --pose_perturb_seed_offset
  fi

  if [[ -n "$CHECKPOINT_RUN" ]]; then
    CMD+=(--checkpoint "$CHECKPOINT_RUN")
  fi

  load_pretrained_weights="$(param LOAD_PRETRAINED_WEIGHTS)"
  if [[ -n "$load_pretrained_weights" ]]; then
    if is_on "$load_pretrained_weights"; then
      CMD+=(--hydra_override "model.model_config.load_pretrained_weights=true")
    else
      CMD+=(--hydra_override "model.model_config.load_pretrained_weights=false")
    fi
  fi

  if ! is_on "$(param LOG_CHUNKS)"; then
    CMD+=(--no_log_chunks)
  fi
  if is_on "$(param KEEP_CHUNK_CACHE)"; then
    CMD+=(--keep_chunk_cache)
  fi

  if is_on "$(param POST_CHUNK_ALIGN)"; then
    CMD+=(--post_chunk_align)
    cmd_arg POST_CHUNK_ALIGN_MODE --post_chunk_align_mode
    cmd_arg POST_CHUNK_ALIGN_MIN_CORR --post_chunk_align_min_corr
    cmd_arg POST_CHUNK_ALIGN_MAX_CORR_PER_VIEW --post_chunk_align_max_corr_per_view
    cmd_arg POST_CHUNK_ALIGN_MAX_CORR --post_chunk_align_max_corr
    cmd_arg POST_CHUNK_ALIGN_SPATIAL_GRID_SIZE --post_chunk_align_spatial_grid_size
    cmd_arg POST_CHUNK_ALIGN_SPATIAL_MIN_CORR_PER_CELL --post_chunk_align_spatial_min_corr_per_cell
    cmd_arg POST_CHUNK_ALIGN_SPATIAL_MAX_CORR_PER_CELL --post_chunk_align_spatial_max_corr_per_cell
    cmd_arg POST_CHUNK_ALIGN_WORKERS --post_chunk_align_workers
    cmd_arg POST_CHUNK_ALIGN_NN_POINTS --post_chunk_align_nn_points
    cmd_arg POST_CHUNK_ALIGN_NN_QUANTILE --post_chunk_align_nn_quantile
    if is_on "$(param POST_CHUNK_ALIGN_NO_SPATIAL_BALANCE)"; then
      CMD+=(--post_chunk_align_no_spatial_balance)
    fi
    if is_on "$(param POST_CHUNK_ALIGN_NN_FALLBACK)"; then
      CMD+=(--post_chunk_align_nn_fallback)
    fi
  fi
  if is_on "$(param COMPUTE_SEAM_ERROR)"; then
    CMD+=(--compute_seam_error)
    cmd_arg SEAM_ERROR_MAX_POINTS_PER_EDGE --seam_error_max_points_per_edge
  fi

  if is_on "$(param EXPORT_TSDF_MESH)"; then
    CMD+=(--export_tsdf_mesh)
    cmd_arg TSDF_OUTPUT_DIR --tsdf_output_dir
    cmd_arg TSDF_VOXEL_SIZE --tsdf_voxel_size
    cmd_arg TSDF_SDF_TRUNC --tsdf_sdf_trunc
    cmd_arg TSDF_DEPTH_TRUNC --tsdf_depth_trunc
    cmd_arg TSDF_MIN_DEPTH --tsdf_min_depth
    cmd_arg TSDF_PIXEL_STRIDE --tsdf_pixel_stride
    cmd_arg TSDF_KEEP_CLUSTERS --tsdf_keep_clusters
    cmd_arg TSDF_MIN_TRIANGLES --tsdf_min_triangles
  fi

  if is_on "$(param BUNDLE_ADJUSTMENT)"; then
    CMD+=(--bundle_adjustment)
    cmd_arg BA_OUTPUT_DIR --ba_output_dir
    cmd_arg BA_MAX_KEYPOINTS --ba_max_keypoints
    cmd_arg BA_PAIR_WINDOW --ba_pair_window
    cmd_arg BA_RATIO_TEST --ba_ratio_test
    cmd_arg BA_MAX_REPROJ_ERROR --ba_max_reproj_error
    if is_on "$(param BA_REFINE_INTRINSICS)"; then
      CMD+=(--ba_refine_intrinsics)
    fi
  fi

  if is_on "$(param RENDER_DOM)"; then
    CMD+=(--render_dom)
    cmd_arg DOM_GSD --dom_gsd
    cmd_arg DOM_AXES --dom_axes
    cmd_arg DOM_UP_AXIS --dom_up_axis
    cmd_arg DOM_SOURCE --dom_source
    cmd_arg DOM_TILE_PX --dom_tile_px
    cmd_arg DOM_MARGIN_PX --dom_margin_px
    cmd_arg DOM_MAX_GAUSSIANS_PER_TILE --dom_max_gaussians_per_tile
    cmd_arg DOM_SPLAT_SCALE --dom_splat_scale
    cmd_arg DOM_DSM_SMOOTH_RADIUS_PX --dom_dsm_smooth_radius_px
    cmd_arg DOM_DSM_SMOOTH_SIGMA --dom_dsm_smooth_sigma
    cmd_arg DOM_DSM_SMOOTH_ITERATIONS --dom_dsm_smooth_iterations
    cmd_arg DOM_DSM_SMOOTH_MIN_WEIGHT --dom_dsm_smooth_min_weight
    if is_on "$(param DOM_SAVE_CONTOURS)"; then
      CMD+=(--dom_save_contours)
    fi
    cmd_arg DOM_OPACITY --dom_opacity
    cmd_arg DOM_GSD_STRIDE --dom_gsd_stride
    cmd_arg DOM_BOUNDS_QMIN --dom_bounds_quantile_min
    cmd_arg DOM_BOUNDS_QMAX --dom_bounds_quantile_max
    cmd_arg DOM_PADDING_M --dom_padding_m
    cmd_arg DOM_MAX_PIXELS --dom_max_pixels
    cmd_arg DOM_EPSG --dom_epsg
    if is_on "$(param DOM_ALLOW_LARGE)"; then
      CMD+=(--dom_allow_large)
    fi
    if ! is_on "$(param DOM_SAVE_TILES)"; then
      CMD+=(--dom_no_save_tiles)
    fi
  fi

  if is_on "$(param GSPLAT_REFINE)"; then
    CMD+=(--gsplat_refine)
    cmd_arg GSPLAT_STEPS --gsplat_steps
    cmd_arg GSPLAT_MAX_GAUSSIANS --gsplat_max_gaussians
    cmd_arg GSPLAT_RENDER_SCALE --gsplat_render_scale
    cmd_arg GSPLAT_BUNDLE_IMAGES --gsplat_bundle_images
    if is_on "$(param GSPLAT_SAVE_RENDERED_VIEWS)"; then
      CMD+=(--gsplat_save_rendered_views)
    fi
  fi

  DRY_RUN_RUN="${DRY_RUN:-$(param_default DRY_RUN 0)}"
  if is_on "$DRY_RUN_RUN"; then
    printf '[DRY_RUN]'
    printf ' %q' "${CMD[@]}"
    printf '\n'
  else
    "${CMD[@]}"
    evaluate_spatial_result "${SCENE_DIR}" "${output_rrd}"
  fi

  echo "<<< Finished scene: $scene_name"
done < <(emit_scene_entries)

echo ""
echo "============================================"
echo "All scenes completed (count=$scene_count)"
echo "Scene list: $SCENE_LIST"
echo "============================================"
