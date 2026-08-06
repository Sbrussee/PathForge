"""Adapters that let LazySlide execute PathForge-native feature extractors."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from pathforge.core.feature_extractors.base import FeatureExtractorBase


def _to_canonical_patch_image(tile: Any) -> np.ndarray:
    """Normalize one LazySlide tile to PathForge's RGB ``uint8`` patch contract.

    Args:
        tile: A LazySlide tile, typically a NumPy array or Torch tensor in HWC
            or CHW layout.

    Returns:
        An RGB NumPy array with shape ``[H, W, 3]`` and ``uint8`` dtype.

    Raises:
        TypeError: If the tile is not a NumPy array or Torch tensor.
        ValueError: If the tile does not have three dimensions or an image-like
            channel axis.
    """
    if isinstance(tile, torch.Tensor):
        tile = tile.detach().cpu().numpy()
    if not isinstance(tile, np.ndarray):
        raise TypeError(
            "LazySlide must provide a NumPy array or Torch tensor tile, "
            f"got {type(tile)!r}."
        )
    if tile.ndim != 3:
        raise ValueError(
            "LazySlide tile must have shape [H, W, C] or [C, H, W], "
            f"got {tile.shape}."
        )

    if tile.shape[0] in (1, 3, 4) and tile.shape[-1] not in (1, 3, 4):
        tile = np.moveaxis(tile, 0, -1)
    if tile.shape[-1] not in (1, 3, 4):
        raise ValueError(
            "LazySlide tile must have one, three, or four channels, "
            f"got shape {tile.shape}."
        )

    if np.issubdtype(tile.dtype, np.floating):
        tile = np.clip(tile, 0, 255).round().astype(np.uint8)
    elif tile.dtype != np.uint8:
        tile = np.clip(tile, 0, 255).astype(np.uint8)

    if tile.shape[-1] == 1:
        tile = np.repeat(tile, 3, axis=-1)
    elif tile.shape[-1] == 4:
        tile = tile[..., :3]
    return tile


class LazySlideFeatureExtractorAdapter:
    """Expose a PathForge extractor through LazySlide's image-model protocol.

    Args:
        name: Stable configured name used by LazySlide for its output feature key.
        extractor: Constructed PathForge-native image feature extractor.

    Example:
        >>> adapter = LazySlideFeatureExtractorAdapter("my_encoder", extractor)
        >>> embeddings = adapter.encode_image(images)
    """

    def __init__(self, name: str, extractor: FeatureExtractorBase) -> None:
        self.name = name
        self._extractor = extractor

    def to(self, device: str) -> LazySlideFeatureExtractorAdapter:
        """Move the wrapped extractor to ``device`` and return this adapter."""
        self._extractor.to(device)
        return self

    def get_transform(self) -> Any:
        """Return a transform accepting LazySlide tiles via the PathForge contract."""
        transform = self._extractor.get_transform()
        if not callable(transform):
            return transform

        def adapt_then_transform(tile: Any) -> Any:
            return transform(_to_canonical_patch_image(tile))

        return adapt_then_transform

    def encode_image(self, images: Tensor) -> Tensor:
        """Encode an image batch with shape ``[B, C, H, W]`` into ``[B, D]``."""
        return self._extractor.encode_images(images)
