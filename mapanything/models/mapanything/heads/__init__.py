# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything heads module.
"""

from mapanything.models.mapanything.heads.semantic_head import (
    SemanticSegmentationHead,
    MultiTaskHead,
)

__all__ = [
    "SemanticSegmentationHead",
    "MultiTaskHead",
]
