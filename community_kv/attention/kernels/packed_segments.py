# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Compact selected-community descriptors for direct segmented attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch

ATTENTION_TILE = 64


@dataclass(frozen=True, slots=True)
class PackedSegments:
    """Device tensors consumed by the fixed-shape native attention operator."""

    communities: torch.Tensor
    cumulative_ends: torch.Tensor
    descriptor_counts: torch.Tensor
    tile_descriptors: torch.Tensor
    token_budget: int
    num_sink: int


__all__ = ["ATTENTION_TILE", "PackedSegments"]
