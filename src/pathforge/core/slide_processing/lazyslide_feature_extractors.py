"""Adapters that let LazySlide execute PathForge-native feature extractors."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from pathforge.core.feature_extractors.base import FeatureExtractorBase


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

    def to(self, device: str) -> "LazySlideFeatureExtractorAdapter":
        """Move the wrapped model to the LazySlide-selected device.

        Args:
            device: Torch device string such as ``"cpu"`` or ``"cuda"``.

        Returns:
            This adapter after delegating device placement to its extractor.

        Example:
            >>> adapter.to("cpu") is adapter
            True
        """
        self._extractor.to(device)
        return self

    def get_transform(self) -> Any:
        """Return preprocessing for one image tile.

        Returns:
            The wrapped extractor's transform, typically mapping an ``HWC``
            image to a ``[C, H, W]`` tensor.

        Example:
            >>> transform = adapter.get_transform()
        """
        return self._extractor.get_transform()

    def encode_image(self, images: Tensor) -> Tensor:
        """Encode a tile batch using the wrapped PathForge extractor.

        Args:
            images: Batch tensor with shape ``[B, C, H, W]``.

        Returns:
            Embedding tensor with shape ``[B, D]``.

        Example:
            >>> embeddings = adapter.encode_image(images)
        """
        return self._extractor.encode_images(images)
