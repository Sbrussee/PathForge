"""PathForge-native extractor for the gated Waiv Mascaret model."""

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

MASCARET_REPOSITORY_ID = "wearewaiv/mascaret"
"""Hugging Face repository containing Mascaret's weights and custom model code."""

MASCARET_EMBEDDING_DIMENSION = 1536
"""Dimension of Mascaret's normalized CLS embeddings."""


@register_feature_extractor("mascaret")
class MascaretFeatureExtractor(FeatureExtractorBase):
    """Extract normalized 1536-dimensional Mascaret CLS embeddings from patches.

    The gated Hugging Face repository uses the standard Hugging Face credential
    lookup: an ``hf auth login`` cache or ``HF_TOKEN`` in the job environment.
    No credentials are accepted through the PathForge configuration.

    Example:
        >>> extractor = MascaretFeatureExtractor()
        >>> features = extractor.encode_images(torch.zeros((2, 3, 224, 224)))
        >>> features.shape
        torch.Size([2, 1536])
    """

    def build_model(self) -> torch.nn.Module:
        """Load Mascaret's current default revision from Hugging Face.

        Returns:
            The custom-code Mascaret model loaded in the Hugging Face user's
            authenticated context.

        Raises:
            RuntimeError: If the repository is gated and the current account
                has not accepted its access terms or is not authenticated.
        """
        try:
            return AutoModel.from_pretrained(
                MASCARET_REPOSITORY_ID,
                trust_remote_code=True,
            )
        except (GatedRepoError, OSError) as error:
            if _is_gated_repository_error(error):
                raise RuntimeError(
                    "Mascaret is a gated Hugging Face model. Accept Mascaret's "
                    "access terms for 'wearewaiv/mascaret' with the executing "
                    "account, then authenticate using 'hf auth login' or set "
                    "HF_TOKEN in the job environment."
                ) from error
            raise

    def get_transform(self) -> Compose:
        """Return Mascaret's config-driven preprocessing for a PathForge RGB patch.

        Returns:
            A transform that accepts a canonical ``uint8`` RGB NumPy patch with
            shape ``[H, W, 3]``, converts it to a PIL image, resizes the short
            side to 224 pixels, center-crops to ``[3, 224, 224]``, and
            normalizes using the loaded model's ``pixel_mean`` and
            ``pixel_std`` values.
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
        """Encode ``[B, 3, 224, 224]`` patches into normalized ``[B, 1536]`` CLS vectors.

        Args:
            images: Preprocessed image batch with shape ``[B, 3, 224, 224]`` on
                the same device as the model.

        Returns:
            L2-normalized CLS embeddings with shape ``[B, 1536]``.

        Raises:
            RuntimeError: If Mascaret returns an unexpected embedding shape.
        """
        with torch.inference_mode():
            embeddings = self.model.encode(images)

        if embeddings.ndim != 2 or embeddings.shape != (
            images.shape[0],
            MASCARET_EMBEDDING_DIMENSION,
        ):
            raise RuntimeError(
                "Mascaret must return normalized CLS embeddings with shape "
                f"[B, {MASCARET_EMBEDDING_DIMENSION}], but returned "
                f"{tuple(embeddings.shape)} for batch size {images.shape[0]}."
            )
        return embeddings


def _is_gated_repository_error(error: GatedRepoError | OSError) -> bool:
    """Return whether a Hugging Face loading error indicates gated access.

    Args:
        error: Error raised while resolving the Hugging Face model.

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
