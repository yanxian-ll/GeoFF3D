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

emit_scene_entries() {
  python3 - "$SCENE_LIST" <<'PY'
import base64
import json
import os
import re
import sys
from pathlib import Path

scene_list = Path(sys.argv[1]).expanduser()

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

raw_lines = []
for raw in scene_list.read_text(encoding="utf-8").splitlines():
    if not raw.strip() or raw.lstrip().startswith("#"):
        continue
    line = strip_inline_comment(raw.rstrip("\n"))
    if line.strip():
        raw_lines.append(line)

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

if not raw_lines:
    sys.exit(0)

first = raw_lines[0].strip()
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
    --max_side "$(param_default MAX_SIDE 518)" \
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
  ALL_IN_ONE_RUNNER_RUN="$(param_required ALL_IN_ONE_RUNNER)"
  ALL_IN_ONE_METHOD_RUN="$(param_required ALL_IN_ONE_METHOD)"
  OUTPUT_BASE_RUN="$(param_default OUTPUT_BASE "$ROOT_DIR/outputs/all_in_one/${METHOD_NAME_RUN}")"
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
  echo "    Backend:    $ALL_IN_ONE_RUNNER_RUN / $ALL_IN_ONE_METHOD_RUN"
  echo "    Scene dir:  $SCENE_DIR"
  echo "    Output RRD: $output_rrd"
  echo "    Params:     stride=$(param_default STRIDE 1) max_side=$(param_default MAX_SIDE 512) size_multiple=$(param_default SIZE_MULTIPLE "$(param_default PATCH_SIZE 16)")"

  if [[ "$ALL_IN_ONE_RUNNER_RUN" == "model" ]]; then
    CMD=(
      python3 "$ROOT_DIR/scripts/run_all_in_one_to_rrd.py"
      --model "$ALL_IN_ONE_METHOD_RUN"
      --machine "$(param_default MACHINE aws)"
      --scene_dir "$SCENE_DIR"
      --output_rrd "$output_rrd"
      --device "cuda:$CUDA_DEVICE_RUN"
      --seed 0
    )
    add_common_io_args
    cmd_arg NORM_TYPE --norm_type
    cmd_arg CHECKPOINT --checkpoint
    cmd_arg PRED_MIN_DEPTH --pred_min_depth
    cmd_arg CONF_QUANTILE --conf_quantile
    cmd_arg MAX_POINTS_PER_VIEW --max_points_per_view
    cmd_arg VOXEL_DOWNSAMPLE --voxel_downsample
    cmd_arg MAX_PRED_POINTS --max_pred_points
    cmd_arg MAX_GT_POINTS --max_gt_points
    cmd_arg SCENE_IO_WORKERS --scene_io_workers
    if ! is_on "$(param_default POINT_DOWNSAMPLE 1)"; then
      CMD+=(--no_point_downsample)
    fi
  else
    echo "[ERROR] Unknown ALL_IN_ONE_RUNNER=$ALL_IN_ONE_RUNNER_RUN" >&2
    exit 2
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
echo "All all-in-one scenes completed (count=$scene_count)"
echo "Scene list: $SCENE_LIST"
echo "============================================"
