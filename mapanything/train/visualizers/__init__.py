from __future__ import annotations

from typing import Any, List

from .base import TensorboardVisualizerList
_VISUALIZER_REGISTRY = {}


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except AttributeError:
        return getattr(cfg, key, default)


def build_tensorboard_visualizers(args, writer=None) -> TensorboardVisualizerList:
    """Build TensorBoard visualizers from args.train_params.visualization.

    Visualizer implementations can be registered in ``_VISUALIZER_REGISTRY``.
    """
    cfg = _cfg_get(args.train_params, "visualization", None)
    enabled = bool(_cfg_get(cfg, "enabled", False))

    if writer is None or not enabled:
        return TensorboardVisualizerList([], cfg=cfg)

    names = _cfg_get(cfg, "visualizers", [])
    if isinstance(names, str):
        names = [names]

    visualizers: List = []
    for name in names:
        if name not in _VISUALIZER_REGISTRY:
            raise ValueError(
                f"Unknown TensorBoard visualizer '{name}'. "
                f"Available: {sorted(_VISUALIZER_REGISTRY.keys())}"
            )
        visualizers.append(_VISUALIZER_REGISTRY[name](cfg=cfg))

    return TensorboardVisualizerList(visualizers, cfg=cfg)
