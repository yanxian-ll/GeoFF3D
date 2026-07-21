#!/bin/bash
set -euo pipefail

export HYDRA_FULL_ERROR=1
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-16}"
export OPENCV_IO_ENABLE_OPENEXR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

if [[ -z "${SCENE_LIST:-}" ]]; then
  method_scene_list="$SCRIPT_DIR/${METHOD_NAME:-}_scenes.yaml"
  if [[ -n "${METHOD_SCENE_LIST:-}" ]]; then
    SCENE_LIST="$METHOD_SCENE_LIST"
  elif [[ -n "${METHOD_NAME:-}" && -f "$method_scene_list" ]]; then
    SCENE_LIST="$method_scene_list"
  else
    SCENE_LIST="$SCRIPT_DIR/../default_scenes.yaml"
  fi
fi
CLI_CUDA_DEVICE="${1:-}"
PARAMS_LIST="${PARAMS_LIST:-$SCRIPT_DIR/default_params.yaml}"

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

OPTIM_RESULT_MISSING=()

require_result_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    OPTIM_RESULT_MISSING+=("$path")
  fi
}

optim_scene_result_complete() {
  local output_rrd="$1"
  optim_scene_prediction_complete "$output_rrd" || return 1
  if is_on "$(param_default RUN_METRICS 0)"; then
    optim_scene_metrics_complete "$output_rrd" || return 1
  fi
  return 0
}

optim_scene_prediction_complete() {
  local output_rrd="$1"
  local result_dir="${output_rrd%.*}"
  local eval_dir="$result_dir/eval"

  OPTIM_RESULT_MISSING=()

  require_result_file "$output_rrd"
  require_result_file "${output_rrd%.*}.json"

  case "$(param_required OPTIM_RUNNER)" in
    streaming|vggt_long|vggt_slam|vggt_slam2*|mast3r_sfm|droid_slam|gaussian)
      require_result_file "$eval_dir/meta.json"
      require_result_file "$eval_dir/pred_cameras.npz"
      require_result_file "$eval_dir/pred_points.ply"
      ;;
  esac

  [[ "${#OPTIM_RESULT_MISSING[@]}" -eq 0 ]]
}

optim_scene_metrics_complete() {
  local output_rrd="$1"
  local result_dir="${output_rrd%.*}"
  local eval_dir="$result_dir/eval"

  OPTIM_RESULT_MISSING=()
  require_result_file "$eval_dir/metrics.json"
  require_result_file "$eval_dir/metrics_summary.csv"

  [[ "${#OPTIM_RESULT_MISSING[@]}" -eq 0 ]]
}

emit_scene_entries() {
  python3 - "$SCENE_LIST" "$PARAMS_LIST" <<'PY'
import base64
import json
import os
import re
import sys
from pathlib import Path

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
    return re.sub(r"[^0-9A-Za-z]+", "_", str(key).strip()).strip("_").upper()

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
    if i >= len(lines) or lines[i][0] < indent:
        return None, i
    is_list = lines[i][0] == indent and lines[i][1].startswith("- ")
    if is_list:
        out = []
        while i < len(lines):
            cur_indent, text = lines[i]
            if cur_indent < indent or cur_indent != indent or not text.startswith("- "):
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
        if cur_indent < indent or cur_indent != indent or text.startswith("- "):
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
    for scene_dir in raw_lines:
        scene_dir = scene_dir.strip()
        emit({"dataset": "default", "scene": Path(scene_dir).name, "scene_dir": scene_dir, "params": {}})
    sys.exit(0)

root_obj, next_i = parse_block(0, lines[0][0])
if next_i < len(lines):
    raise SystemExit(f"Could not parse scene list near line: {raw_lines[next_i]!r}")
if isinstance(root_obj, list):
    for item in root_obj:
        if isinstance(item, str):
            emit({"dataset": "default", "scene": Path(item).name, "scene_dir": item, "params": {}})
    sys.exit(0)
if not isinstance(root_obj, dict):
    raise SystemExit("Scene list must be a path list or a mapping with datasets.")

top_params = params_from(root_obj)
datasets = root_obj.get("datasets", [])
if isinstance(datasets, dict):
    datasets = [{"name": name, **(cfg if isinstance(cfg, dict) else {"root": cfg})} for name, cfg in datasets.items()]
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
        scenes = [{"name": name, **(cfg if isinstance(cfg, dict) else {"path": cfg})} for name, cfg in scenes.items()]
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
        emit({"dataset": dataset_name, "scene": scene_name, "scene_dir": scene_path, "defaults": top_params, "params": params})
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
  if [[ ! -f "${eval_dir}/pred_cameras.npz" || ! -f "${eval_dir}/pred_points.ply" ]]; then
    echo "[METRICS][WARN] Missing eval outputs under ${eval_dir}, skip."
    return 0
  fi

  python "$ROOT_DIR/scripts/evaluate_aligned_reconstruction_metrics.py" \
    --scene_dir "${scene_dir}" \
    --eval_dir "${eval_dir}" \
    --output_json "${eval_dir}/metrics.json" \
    --output_csv "${eval_dir}/metrics_summary.csv" \
    --images_dir "$(param_default IMAGES_DIR images)" \
    --cams_dir "$(param_default CAMS_DIR cams)" \
    --depth_dir "$(param_default DEPTH_DIR depth)" \
    --frame_glob "$(param_default FRAME_GLOB "*")" \
    --num_views "$(param_default NUM_VIEWS 0)" \
    --start "$(param_default START 0)" \
    --stride "$(param_default STRIDE 1)" \
    --max_side "$(param_default MAX_SIDE 518)" \
    --size_multiple "$(param_default SIZE_MULTIPLE "$(param_default PATCH_SIZE 14)")" \
    --depth_scale "$(param_default DEPTH_SCALE 1.0)" \
    --depth_min "$(param_default DEPTH_MIN 1e-6)" \
    --depth_max "$(param_default DEPTH_MAX 1e6)" \
    --thresholds "$(param_default METRIC_THRESHOLDS 0.1,0.2,0.5,1.0)" \
    --rpe_steps "$(param_default METRIC_RPE_STEPS 1,5,10)" \
    --max_gt_points "$(param_default METRIC_MAX_GT_POINTS 300000)" \
    --max_pred_points_eval "$(param_default METRIC_MAX_PRED_POINTS_EVAL 300000)" \
    --max_gt_points_eval "$(param_default METRIC_MAX_GT_POINTS_EVAL 300000)" \
    --max_align_points "$(param_default METRIC_MAX_ALIGN_POINTS 50000)" \
    --icp_iterations "$(param_default METRIC_ICP_ITERATIONS 20)" \
    --icp_trim_quantile "$(param_default METRIC_ICP_TRIM_QUANTILE 0.85)" \
    --gt_io_workers "$(param_default METRIC_GT_IO_WORKERS "$(param_default SCENE_IO_WORKERS 0)")" \
    --seed 0
}

add_common_io_args() {
  cmd_arg NUM_VIEWS --num_views
  cmd_arg STRIDE --stride
  cmd_arg IMAGES_DIR --images_dir
  cmd_arg CAMS_DIR --cams_dir
  cmd_arg DEPTH_DIR --depth_dir
  cmd_arg DEPTH_SCALE --depth_scale
  cmd_arg DEPTH_MIN --depth_min
  cmd_arg DEPTH_MAX --depth_max
  cmd_arg MAX_SIDE --max_side
  local size_multiple
  size_multiple="$(param SIZE_MULTIPLE)"
  if [[ -z "$size_multiple" ]]; then
    size_multiple="$(param PATCH_SIZE)"
  fi
  if [[ -n "$size_multiple" ]]; then
    CMD+=(--size_multiple "$size_multiple")
  fi
}

scene_count=0
while IFS= read -r scene_entry_b64; do
  [[ -z "$scene_entry_b64" ]] && continue
  load_scene_entry "$scene_entry_b64"
  scene_count=$((scene_count + 1))

  METHOD_NAME_RUN="$(param_required METHOD_NAME)"
  OPTIM_RUNNER_RUN="$(param_required OPTIM_RUNNER)"
  OPTIM_METHOD_RUN="$(param_required OPTIM_METHOD)"
  OUTPUT_BASE_RUN="$(param_default OUTPUT_BASE "$ROOT_DIR/outputs/optim/${METHOD_NAME_RUN}")"
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

  echo ""
  echo ">>> Processing scene: $DATASET_NAME/$scene_name"
  echo "    Method:     $METHOD_NAME_RUN"
  echo "    Backend:    $OPTIM_RUNNER_RUN / $OPTIM_METHOD_RUN"
  echo "    Scene dir:  $SCENE_DIR"
  echo "    Output RRD: $output_rrd"
  echo "    Params:     stride=$(param_default STRIDE 1) max_side=$(param_default MAX_SIDE 518) size_multiple=$(param_default SIZE_MULTIPLE "$(param_default PATCH_SIZE 14)")"

  OVERWRITE_RUN="$(param_default OVERWRITE 0)"
  DRY_RUN_RUN="${DRY_RUN:-$(param_default DRY_RUN 0)}"
  if ! is_on "$OVERWRITE_RUN"; then
    if optim_scene_prediction_complete "$output_rrd"; then
      if is_on "$(param_default RUN_METRICS 0)"; then
        if optim_scene_metrics_complete "$output_rrd"; then
          echo "    Status:     SKIP"
          echo "    Reason:     prediction and metrics already exist. Use OVERWRITE=1 to rerun."
          continue
        fi
        echo "    Status:     METRICS"
        echo "    Reason:     prediction exists; only metrics are missing."
        if [[ "${#OPTIM_RESULT_MISSING[@]}" -gt 0 ]]; then
          echo "    Missing:    ${OPTIM_RESULT_MISSING[0]}"
        fi
        if is_on "$DRY_RUN_RUN"; then
          echo "[DRY_RUN] evaluate metrics for ${output_rrd%.*}/eval"
        else
          evaluate_spatial_result "${SCENE_DIR}" "${output_rrd}"
        fi
        echo "<<< Finished scene: $scene_name"
        continue
      fi
      echo "    Status:     SKIP"
      echo "    Reason:     prediction already exists and RUN_METRICS=0. Use OVERWRITE=1 to rerun."
      continue
    fi
    if [[ "${#OPTIM_RESULT_MISSING[@]}" -gt 0 ]]; then
      echo "    Status:     RUN"
      echo "    Missing:    ${OPTIM_RESULT_MISSING[0]}"
    fi
  fi

  if [[ "$OPTIM_RUNNER_RUN" == "gaussian" ]]; then
    CMD=(
      python3 "$ROOT_DIR/scripts/run_gaussian_slam_to_rrd.py"
      --method "$OPTIM_METHOD_RUN"
      --scene_dir "$SCENE_DIR"
      --output_rrd "$output_rrd"
      --device "cuda:$CUDA_DEVICE_RUN"
      --seed 0
    )
    add_common_io_args
    cmd_arg ARTDECO_CONFIG --artdeco_config
    cmd_arg ARTDECO_CHECKPOINT --artdeco_checkpoint
  elif [[ "$OPTIM_RUNNER_RUN" == "vggt_slam" ]]; then
    CMD=(
      "$(param_default VGGT_SLAM_PYTHON python3)" "$ROOT_DIR/third_party/vggt-slam/run_scene_to_rrd.py"
      --method "$OPTIM_METHOD_RUN"
      --scene_dir "$SCENE_DIR"
      --output_rrd "$output_rrd"
      --device "cuda:$CUDA_DEVICE_RUN"
      --seed 0
    )
    add_common_io_args
    cmd_arg SUBMAP_SIZE --submap_size
    cmd_arg OVERLAPPING_WINDOW_SIZE --overlapping_window_size
    cmd_arg VGGT_MODEL_PATH --vggt_model_path
    cmd_arg MAX_LOOPS --max_loops
    cmd_arg MIN_DISPARITY --min_disparity
    cmd_arg MAX_PRED_POINTS --max_pred_points
    cmd_arg MAX_GT_POINTS --max_gt_points
    cmd_arg MAX_POINTS_PER_VIEW --max_points_per_view
    cmd_arg VOXEL_DOWNSAMPLE --voxel_downsample
    cmd_arg GLOBAL_POINT_STRIDE --global_point_stride

    if ! is_on "$(param_default POINT_DOWNSAMPLE 1)"; then
      CMD+=(--no_point_downsample)
    fi

    if is_on "$(param_default KEEP_INTERMEDIATE 0)"; then
      CMD+=(--keep_intermediate)
    fi

    if is_on "$(param DISABLE_KEYFRAME_SELECTION)"; then
      CMD+=(--disable_keyframe_selection)
    fi
    if is_on "$(param USE_SIM3)"; then
      CMD+=(--use_sim3)
    fi
  elif [[ "$OPTIM_RUNNER_RUN" == "vggt_long" ]]; then
    CMD=(
      python3 "$ROOT_DIR/scripts/run_vggt_slam_to_rrd.py"
      --method "$OPTIM_METHOD_RUN"
      --scene_dir "$SCENE_DIR"
      --output_rrd "$output_rrd"
      --device "cuda:$CUDA_DEVICE_RUN"
      --seed 0
    )
    add_common_io_args

    cmd_arg VGGT_LONG_CONFIG --vggt_long_config
    cmd_arg VGGT_MODEL_PATH --vggt_model_path
    cmd_arg MAX_PRED_POINTS --max_pred_points
    cmd_arg MAX_GT_POINTS --max_gt_points
    cmd_arg MAX_POINTS_PER_VIEW --max_points_per_view
    cmd_arg VOXEL_DOWNSAMPLE --voxel_downsample
    cmd_arg SCENE_IO_WORKERS --scene_io_workers

    if ! is_on "$(param_default POINT_DOWNSAMPLE 1)"; then
      CMD+=(--no_point_downsample)
    fi

    if is_on "$(param_default KEEP_INTERMEDIATE 0)"; then
      CMD+=(--keep_intermediate)
    fi
  elif [[ "$OPTIM_RUNNER_RUN" == "streaming" ]]; then
    CMD=(
      python3 "$ROOT_DIR/scripts/run_streaming_to_rrd.py"
      --method "$OPTIM_METHOD_RUN"
      --scene_dir "$SCENE_DIR"
      --output_rrd "$output_rrd"
      --device "cuda:$CUDA_DEVICE_RUN"
      --seed 0
    )
    add_common_io_args

    cmd_arg STREAM_PYTHON --python
    cmd_arg STREAM_MODEL_PATH --model_path
    cmd_arg STREAM_MODE --stream_mode
    cmd_arg LINGBOT_MODEL_PATH --model_path
    cmd_arg LINGBOT_STREAM_MODE --stream_mode
    cmd_arg LINGBOT_KEYFRAME_INTERVAL --keyframe_interval
    cmd_arg TTT3R_RESET_INTERVAL --reset_interval
    cmd_arg TTT3R_MODEL_UPDATE_TYPE --model_update_type
    cmd_arg MAX_PRED_POINTS --max_pred_points
    cmd_arg MAX_GT_POINTS --max_gt_points
    cmd_arg MAX_POINTS_PER_VIEW --max_points_per_view
    cmd_arg VOXEL_DOWNSAMPLE --voxel_downsample
    cmd_arg SCENE_IO_WORKERS --scene_io_workers

    USE_SDPA_RUN="$(param USE_SDPA)"
    if [[ -n "$USE_SDPA_RUN" ]]; then
      if is_on "$USE_SDPA_RUN"; then
        CMD+=(--use_sdpa)
      else
        CMD+=(--no-use_sdpa)
      fi
    fi

    if ! is_on "$(param_default POINT_DOWNSAMPLE 1)"; then
      CMD+=(--no_point_downsample)
    fi

    if is_on "$(param_default KEEP_INTERMEDIATE 0)"; then
      CMD+=(--keep_intermediate)
    fi
  elif [[ "$OPTIM_RUNNER_RUN" == vggt_slam2* ]]; then
    CMD=(
      "$(param_default VGGT_SLAM2_PYTHON python3)" "$ROOT_DIR/third_party/vggt-slam2.0/run_scene_to_rrd.py"
      --method "$OPTIM_METHOD_RUN"
      --scene_dir "$SCENE_DIR"
      --output_rrd "$output_rrd"
      --device "cuda:$CUDA_DEVICE_RUN"
      --seed 0
    )
    add_common_io_args

    cmd_arg SUBMAP_SIZE --submap_size
    cmd_arg OVERLAPPING_WINDOW_SIZE --overlapping_window_size
    cmd_arg VGGT_MODEL_PATH --vggt_model_path
    cmd_arg MAX_LOOPS --max_loops
    cmd_arg MIN_DISPARITY --min_disparity
    cmd_arg CONF_THRESHOLD --conf_threshold
    cmd_arg LC_THRES --lc_thres
    cmd_arg VIS_VOXEL_SIZE --vis_voxel_size

    cmd_arg MAX_PRED_POINTS --max_pred_points
    cmd_arg MAX_GT_POINTS --max_gt_points
    cmd_arg MAX_POINTS_PER_VIEW --max_points_per_view
    cmd_arg VOXEL_DOWNSAMPLE --voxel_downsample
    cmd_arg GLOBAL_POINT_STRIDE --global_point_stride

    if ! is_on "$(param_default POINT_DOWNSAMPLE 1)"; then
      CMD+=(--no_point_downsample)
    fi

    if is_on "$(param_default KEEP_INTERMEDIATE 0)"; then
      CMD+=(--keep_intermediate)
    fi

    if is_on "$(param DISABLE_KEYFRAME_SELECTION)"; then
      CMD+=(--disable_keyframe_selection)
    fi
  elif [[ "$OPTIM_RUNNER_RUN" == "mast3r_sfm" ]]; then
    CMD=(
      "$(param_default MAST3R_PYTHON python3)" "$ROOT_DIR/scripts/run_mast3r_sfm_to_rrd.py"
      --scene_dir "$SCENE_DIR"
      --output_rrd "$output_rrd"
      --device "cuda:$CUDA_DEVICE_RUN"
      --seed 0
    )
    add_common_io_args
    cmd_arg MAST3R_MODEL_PATH --model_path
    cmd_arg MAST3R_RETRIEVAL_MODEL --retrieval_model
    cmd_arg MAST3R_SCENE_GRAPH --scene_graph
    cmd_arg MAST3R_PREFILTER --prefilter
    cmd_arg MAX_PRED_POINTS --max_pred_points
    cmd_arg MAX_GT_POINTS --max_gt_points
    if ! is_on "$(param_default MAST3R_SYMMETRIZE 1)"; then
      CMD+=(--no_symmetrize)
    fi
  elif [[ "$OPTIM_RUNNER_RUN" == "droid_slam" ]]; then
    CMD=(
      python3 "$ROOT_DIR/scripts/run_droid_slam_to_rrd.py"
      --scene_dir "$SCENE_DIR"
      --output_rrd "$output_rrd"
      --device "cuda:$CUDA_DEVICE_RUN"
      --seed 0
    )
    add_common_io_args
    cmd_arg DROID_PYTHON --python
    cmd_arg DROID_ROOT --droid_root
    cmd_arg DROID_WEIGHTS --weights
    cmd_arg DROID_BUFFER --buffer
    cmd_arg DROID_WARMUP --warmup
    cmd_arg DROID_FILTER_THRESH --filter_thresh
    cmd_arg DROID_KEYFRAME_THRESH --keyframe_thresh
    cmd_arg DROID_FRONTEND_THRESH --frontend_thresh
    cmd_arg DROID_FRONTEND_WINDOW --frontend_window
    cmd_arg DROID_FRONTEND_RADIUS --frontend_radius
    cmd_arg DROID_FRONTEND_NMS --frontend_nms
    cmd_arg DROID_BACKEND_THRESH --backend_thresh
    cmd_arg DROID_BACKEND_RADIUS --backend_radius
    cmd_arg DROID_BACKEND_NMS --backend_nms
    cmd_arg DROID_MIN_DISP_RATIO --min_disp_ratio
    cmd_arg MAX_PRED_POINTS --max_pred_points
    cmd_arg MAX_GT_POINTS --max_gt_points
    cmd_arg SCENE_IO_WORKERS --scene_io_workers
    if is_on "$(param_default DROID_ASYNCHRONOUS 0)"; then
      CMD+=(--asynchronous)
    fi
    if is_on "$(param_default DROID_UPSAMPLE 1)"; then
      CMD+=(--upsample)
    fi
  else
    echo "[ERROR] Unknown OPTIM_RUNNER=$OPTIM_RUNNER_RUN" >&2
    exit 2
  fi

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
