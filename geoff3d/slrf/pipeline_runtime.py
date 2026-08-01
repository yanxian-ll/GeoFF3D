# -*- coding: utf-8 -*-
"""Runtime logging, progress, and scene-manifest filtering helpers."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
class _TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return bool(self.streams and getattr(self.streams[0], "isatty", lambda: False)())


class RunLogger:
    def __init__(self, output_path: Path, log_file: Optional[str], enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.path: Optional[Path] = None
        self._fh = None
        self._old_stdout = None
        self._old_stderr = None
        if not self.enabled:
            return
        if log_file:
            self.path = Path(log_file).expanduser().resolve()
        else:
            self.path = Path(output_path).expanduser().resolve() / "logs" / "pipeline.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def install(self) -> None:
        if not self.enabled or self.path is None:
            return
        self._fh = self.path.open("w", encoding="utf-8", buffering=1)
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = _TeeStream(sys.stdout, self._fh)  # type: ignore[assignment]
        sys.stderr = _TeeStream(sys.stderr, self._fh)  # type: ignore[assignment]
        print(f"[LOG] Saving full run log to: {self.path}")

    def close(self) -> None:
        if self._fh is None:
            return
        if self._old_stdout is not None:
            sys.stdout = self._old_stdout
        if self._old_stderr is not None:
            sys.stderr = self._old_stderr
        self._fh.close()
        self._fh = None

    def detail(self, message: str) -> None:
        if self._fh is None:
            return
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._fh.write(f"[{stamp}] {message}\n")
        self._fh.flush()


def stage(name: str, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"\n[STAGE] {name}{suffix}")


def iter_progress(
    values: Iterable,
    *,
    desc: str,
    total: Optional[int] = None,
    enabled: bool = True,
):
    if enabled and tqdm is not None:
        return tqdm(
            values,
            desc=desc,
            total=total,
            unit="chunk",
            dynamic_ncols=True,
            file=sys.__stderr__,
        )
    return values


def _pose_filter_reason(meta: Dict[str, object], stem: str) -> Optional[str]:
    cams = meta.get("cams", {})
    cam = cams.get(stem) if isinstance(cams, dict) else None
    if cam is None:
        return "missing_camera_prior"
    T = np.asarray(cam.get("T_c2w", None), dtype=np.float64)
    if T.shape != (4, 4):
        return f"invalid_T_c2w_shape:{T.shape}"
    if not np.isfinite(T).all():
        return "non_finite_T_c2w"
    return None


def filter_views_meta_to_valid_poses(
    views: Sequence[Dict[str, object]],
    meta: Dict[str, object],
    *,
    output_path: Path,
    run_logger: RunLogger,
) -> Tuple[List[Dict[str, object]], Dict[str, object], List[Dict[str, object]]]:
    ignored: List[Dict[str, object]] = []
    keep_stems: List[str] = []
    for stem in meta.get("stems", []):
        stem = str(stem)
        reason = _pose_filter_reason(meta, stem)
        if reason is None:
            keep_stems.append(stem)
        else:
            ignored.append(
                {
                    "stem": stem,
                    "reason": reason,
                    "image_path": str(meta.get("image_paths", {}).get(stem, "")),
                    "cam_path": str(meta.get("cam_paths", {}).get(stem, "")),
                    "depth_path": str(meta.get("depth_paths", {}).get(stem, "")),
                }
            )

    if not keep_stems:
        preview = ", ".join(str(item["stem"]) for item in ignored[:8])
        raise RuntimeError(
            "No selected frames have valid camera poses after filtering. "
            f"First ignored frames: {preview}"
        )

    keep_set = set(keep_stems)
    filtered_views = [
        view for view in views if str(view.get("stem", "")) in keep_set
    ]
    filtered_meta = dict(meta)
    filtered_meta["stems"] = keep_stems

    for key in ("image_paths", "depth_paths", "cam_paths", "cams"):
        value = meta.get(key, {})
        if isinstance(value, dict):
            filtered_meta[key] = {
                stem: value[stem] for stem in keep_stems if stem in value
            }

    cams = filtered_meta.get("cams", {})
    depths = filtered_meta.get("depth_paths", {})
    filtered_meta["num_cam_priors"] = int(
        sum(1 for stem in keep_stems if isinstance(cams, dict) and stem in cams)
    )
    filtered_meta["num_depth_priors"] = int(
        sum(1 for stem in keep_stems if isinstance(depths, dict) and stem in depths)
    )
    filtered_meta["ignored_no_pose_frames"] = ignored
    filtered_meta["num_ignored_no_pose_frames"] = int(len(ignored))

    if ignored:
        log_dir = (
            run_logger.path.parent
            if run_logger.path is not None
            else Path(output_path).expanduser().resolve() / "logs"
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        json_path = log_dir / "ignored_no_pose_frames.json"
        txt_path = log_dir / "ignored_no_pose_frames.txt"
        json_path.write_text(
            json.dumps(ignored, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        txt_path.write_text(
            "\n".join(
                f"{item['stem']}\t{item['reason']}\t{item.get('image_path', '')}\t{item.get('cam_path', '')}"
                for item in ignored
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            "[WARN] Ignored frames without valid poses: "
            f"{len(ignored)}/{len(meta.get('stems', []))}; "
            f"kept={len(keep_stems)}. See {json_path}"
        )
        for item in ignored:
            run_logger.detail(
                "[ignored_no_pose] "
                f"stem={item['stem']}, reason={item['reason']}, "
                f"image={item.get('image_path', '')}, cam={item.get('cam_path', '')}"
            )

    return filtered_views, filtered_meta, ignored

