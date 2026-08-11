"""PathForge-native extractor for the gated Waiv Phaet model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from huggingface_hub.errors import GatedRepoError
from torch import Tensor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToPILImage,
    ToTensor,
)
from transformers import AutoModel

from pathforge.core.feature_extractors.base import FeatureExtractorBase
from pathforge.core.feature_extractors.factory import register_feature_extractor

PHAET_REPOSITORY_ID = "wearewaiv/phaet"
"""Hugging Face repository containing Phaet weights and custom model code."""

PHAET_EMBEDDING_DIMENSION = 1024
"""Dimension of Phaet's normalized CLS embeddings."""


@register_feature_extractor("phaet")
class PhaetFeatureExtractor(FeatureExtractorBase):
    """Extract normalized 1024-dimensional Phaet CLS embeddings from patches.

    The gated Hugging Face repository uses the standard Hugging Face credential
    lookup: an ``hf auth login`` cache or ``HF_TOKEN`` in the job environment.
    No credentials are accepted through the PathForge configuration.

    Example:
        >>> extractor = PhaetFeatureExtractor()
        >>> features = extractor.encode_images(torch.zeros((2, 3, 224, 224)))
        >>> features.shape
        torch.Size([2, 1024])
    """

    def build_model(self) -> torch.nn.Module:
        """Load Phaet's current default revision from Hugging Face.

        Returns:
            The custom-code Phaet model loaded in the authenticated Hugging
            Face context.

        Raises:
            RuntimeError: If the repository is gated and the current account
                has not accepted its access terms or is not authenticated.
        """
        try:
            return AutoModel.from_pretrained(
                PHAET_REPOSITORY_ID,
                trust_remote_code=True,
            )
        except (GatedRepoError, OSError) as error:
            if _is_gated_repository_error(error):
                raise RuntimeError(
                    "Phaet is a gated Hugging Face model. Accept Phaet's "
                    "access terms for 'wearewaiv/phaet' with the executing "
                    "account, then authenticate using 'hf auth login' or set "
                    "HF_TOKEN in the job environment."
                ) from error
            raise

    def get_transform(self) -> Compose:
        """Return Phaet preprocessing for a canonical PathForge RGB patch.

        Returns:
            A transform that accepts a ``uint8`` RGB NumPy patch with shape
            ``[H, W, 3]``, resizes its short side to 224 pixels, center-crops
            to ``[3, 224, 224]``, and applies Phaet's config normalization.
        """
        config = self.model.config
        return Compose(
            [
                ToPILImage(),
                Resize(224),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=config.pixel_mean, std=config.pixel_std),
            ]
        )

    def encode_images(self, images: Tensor) -> Tensor:
        """Encode ``[B, 3, 224, 224]`` patches into normalized CLS vectors.

        Args:
            images: Preprocessed image batch with shape ``[B, 3, 224, 224]``.

        Returns:
            L2-normalized CLS embeddings with shape ``[B, 1024]``.

        Raises:
            RuntimeError: If Phaet returns an unexpected embedding shape.
        """
        with torch.inference_mode():
            embeddings = self.model.encode(images)

        if embeddings.ndim != 2 or embeddings.shape != (
            images.shape[0],
            PHAET_EMBEDDING_DIMENSION,
        ):
            raise RuntimeError(
                "Phaet must return normalized CLS embeddings with shape "
                f"[B, {PHAET_EMBEDDING_DIMENSION}], but returned "
                f"{tuple(embeddings.shape)} for batch size {images.shape[0]}."
            )
        return embeddings


def _is_gated_repository_error(error: GatedRepoError | OSError) -> bool:
    """Return whether one Hub loading error indicates gated access.

    Args:
        error: Error raised while resolving the model repository.

    Returns:
        ``True`` only for common Hub authentication or gated-repository errors.
    """
    message = str(error).lower()
    markers: Sequence[str] = (
        "gated",
        "access to model",
        "repository not found",
        "401",
        "403",
    )
    return any(marker in message for marker in markers)
