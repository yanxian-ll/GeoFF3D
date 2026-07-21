"""TTT3R streaming wrapper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch


class TTT3RWrapper(torch.nn.Module):
    def __init__(
        self,
        name: str = "ttt3r",
        torch_hub_force_reload: bool = False,
        checkpoint: Optional[str] = None,
        model_path: Optional[str] = None,
        device: str = "cuda",
        image_size: int = 512,
        model_update_type: str = "ttt3r",
        reset_interval: int = 1000000,
        max_points: int = 800000,
        seed: int = 0,
        **_: object,
    ) -> None:
        super().__init__()
        self.name = name
        self.torch_hub_force_reload = torch_hub_force_reload
        self.checkpoint = checkpoint or model_path or "checkpoints/ttt3r/cut3r_512_dpt_4_64.pth"
        self.device_name = device
        self.image_size = int(image_size)
        self.model_update_type = model_update_type
        self.reset_interval = int(reset_interval)
        self.max_points = int(max_points)
        self.seed = int(seed)

    @property
    def display_name(self) -> str:
        return "TTT3R"

    def run(
        self,
        *,
        scene_dir: str | Path,
        image_dir: str | Path,
        output_dir: str | Path,
        pose_log_path: str | Path,
        point_cloud_path: str | Path,
        timing_path: str | Path,
        device: Optional[str] = None,
        python: Optional[str] = None,
        max_points: Optional[int] = None,
        extra_args: Sequence[str] = (),
    ) -> Dict[str, object]:
        del scene_dir
        repo_root = Path(__file__).resolve().parents[4] / "third_party" / "TTT3R"
        runner = Path(__file__).with_name("infer.py")
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_log_path = output_dir / "ttt3r_stdout.log"
        py = python or sys.executable
        use_device = str(device or self.device_name)
        n_points = int(max_points if max_points is not None else self.max_points)
        checkpoint = Path(str(self.checkpoint)).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = Path.cwd() / checkpoint
        cmd = [
            py,
            str(runner),
            "--repo_root",
            str(repo_root),
            "--image_dir",
            str(Path(image_dir).expanduser().resolve()),
            "--output_dir",
            str(output_dir),
            "--pose_log_path",
            str(Path(pose_log_path).expanduser().resolve()),
            "--point_cloud_path",
            str(Path(point_cloud_path).expanduser().resolve()),
            "--timing_path",
            str(Path(timing_path).expanduser().resolve()),
            "--checkpoint",
            str(checkpoint),
            "--device",
            use_device,
            "--image_size",
            str(self.image_size),
            "--model_update_type",
            str(self.model_update_type),
            "--reset_interval",
            str(self.reset_interval),
            "--max_points",
            str(n_points),
            "--seed",
            str(self.seed),
        ]
        cmd.extend(str(x) for x in extra_args)

        with stdout_log_path.open("w", encoding="utf-8") as log_f:
            proc = subprocess.run(cmd, cwd=str(repo_root), stdout=log_f, stderr=subprocess.STDOUT)

        return {
            "return_code": int(proc.returncode),
            "command": cmd,
            "cwd": str(repo_root),
            "stdout_log_path": stdout_log_path,
            "pose_log_path": Path(pose_log_path),
            "point_cloud_path": Path(point_cloud_path),
            "timing_path": Path(timing_path),
            "staged_outputs": {
                "pose_source": str(pose_log_path),
                "point_cloud_path": str(point_cloud_path),
            },
        }

    def run_scene(self, **kwargs: object) -> Dict[str, object]:
        return self.run(**kwargs)


__all__ = ["TTT3RWrapper"]
