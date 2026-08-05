"""Base interface for PathForge-native feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor


class FeatureExtractorBase(ABC):
    """Base class for PathForge patch feature extractors.

    A feature extractor owns its underlying model and defines:

    - how the model is constructed;
    - which preprocessing transform is applied to each patch;
    - how a batch of patches is converted into feature embeddings.

    Subclasses should set any attributes needed by ``build_model`` before
    calling ``super().__init__()``.

    """

    model: Any

    def __init__(self) -> None:
        """Construct the underlying feature-extraction model."""
        self.model = self.build_model()

        if self.model is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.build_model() returned None."
            )

        self.eval()

    @abstractmethod
    def build_model(self) -> Any:
        """Create and return the underlying feature-extraction model.

        Returns:
            The model used to encode image patches. This will commonly be a
            ``torch.nn.Module``, but another model type may be returned when a
            backend supports it.
        """
        raise NotImplementedError

    @abstractmethod
    def get_transform(self) -> Any:
        """Return preprocessing applied to individual image patches.

        The transform should convert one input patch into the representation
        expected by the model, usually a ``torch.Tensor`` with shape
        ``[C, H, W]``.

        Returns:
            A callable transform, or ``None`` when no preprocessing is needed.
        """
        raise NotImplementedError

    @abstractmethod
    def encode_images(self, images: Tensor) -> Tensor:
        """Encode a batch of image patches.

        Args:
            images: Batch tensor with shape ``[B, C, H, W]``.

        Returns:
            Feature tensor with shape ``[B, D]``, where ``D`` is the embedding
            dimension.

        Implementations must preserve the batch dimension so that each input
        patch corresponds to exactly one output feature vector.
        """
        raise NotImplementedError

    def __call__(self, images: Tensor) -> Tensor:
        """Encode a batch of image patches."""
        return self.encode_images(images)

    def to(self, device: str | torch.device) -> FeatureExtractorBase:
        """Move the underlying model to a device when supported."""
        to_method = getattr(self.model, "to", None)

        if callable(to_method):
            moved_model = to_method(device)

            # Most PyTorch models return themselves, but supporting the return
            # value also accommodates model wrappers with immutable semantics.
            if moved_model is not None:
                self.model = moved_model

        return self

    def eval(self) -> FeatureExtractorBase:
        """Put the underlying model in evaluation mode when supported."""
        eval_method = getattr(self.model, "eval", None)

        if callable(eval_method):
            eval_method()

        return self