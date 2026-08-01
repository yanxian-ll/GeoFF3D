"""Standalone wrappers for the streaming benchmark methods."""

from __future__ import annotations

from typing import Type

import torch

from .lingbot_map import LingBotMapWrapper
from .stream3r import STream3RWrapper
from .streamvggt import StreamVGGTWrapper
from .ttt3r import TTT3RWrapper


STREAMING_METHODS: dict[str, Type[torch.nn.Module]] = {
    "lingbot-map": LingBotMapWrapper,
    "stream3r": STream3RWrapper,
    "streamvggt": StreamVGGTWrapper,
    "ttt3r": TTT3RWrapper,
}


def create_streaming_method(name: str, **kwargs: object) -> torch.nn.Module:
    try:
        wrapper = STREAMING_METHODS[str(name)]
    except KeyError as error:
        supported = ", ".join(sorted(STREAMING_METHODS))
        raise ValueError(f"Unknown streaming method {name!r}; choose one of: {supported}") from error
    return wrapper(**kwargs)


__all__ = ["STREAMING_METHODS", "create_streaming_method"]
