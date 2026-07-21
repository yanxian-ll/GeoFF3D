from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except AttributeError:
        return getattr(cfg, key, default)


@dataclass
class VisualizationContext:
    """Runtime context passed from the training loop to visualizers."""

    stage: str  # "train" or "eval"
    prefix: str
    step: int
    epoch: float
    data_iter_step: int


class TensorboardVisualizer:
    """Base class for TensorBoard visualizers.

    A visualizer should:
      1. Decide whether it can handle the current model output.
      2. Log images/scalars without changing tensors or training state.
      3. Detach tensors before CPU/TensorBoard conversion.
    """

    name: str = "base"

    def __init__(self, cfg: Any = None):
        self.cfg = cfg

    def supports(self, result: Dict[str, Any]) -> bool:
        return True

    @torch.no_grad()
    def log(self, writer, result: Dict[str, Any], context: VisualizationContext) -> None:
        raise NotImplementedError


class TensorboardVisualizerList:
    """Container that handles scheduling and dispatching multiple visualizers."""

    def __init__(self, visualizers: List[TensorboardVisualizer], cfg: Any = None):
        self.visualizers = visualizers
        self.cfg = cfg

        self.enabled = bool(cfg_get(cfg, "enabled", False))
        self.train_interval = int(cfg_get(cfg, "train_interval", 0))
        self.eval_interval_epochs = int(cfg_get(cfg, "eval_interval_epochs", 1))
        self.eval_first_batch_only = bool(cfg_get(cfg, "eval_first_batch_only", True))

    def is_enabled(self) -> bool:
        return self.enabled and len(self.visualizers) > 0

    def should_log(
        self,
        stage: str,
        step: int,
        epoch: float,
        data_iter_step: int,
    ) -> bool:
        if not self.is_enabled():
            return False

        if stage == "train":
            return self.train_interval > 0 and step % self.train_interval == 0

        if stage == "eval":
            if self.eval_first_batch_only and data_iter_step != 0:
                return False
            if self.eval_interval_epochs <= 0:
                return True
            return int(epoch) % self.eval_interval_epochs == 0

        return False

    @torch.no_grad()
    def log(
        self,
        writer,
        result: Dict[str, Any],
        context: VisualizationContext,
    ) -> None:
        if writer is None or not self.is_enabled():
            return

        for visualizer in self.visualizers:
            if visualizer.supports(result):
                visualizer.log(writer, result, context)