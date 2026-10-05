"""Unchanged slide-level feature retrieval representations."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from pathforge.slide_retrieval.representation_strategies.base import (
    BaseRetrievalRepresentationStrategy,
)
from pathforge.slide_retrieval.representation_strategies.registry import (
    register_representation_strategy,
)
from pathforge.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)


@register_representation_strategy("slide_features")
class SlideFeatures(BaseRetrievalRepresentationStrategy):
    """Validate and retain one global feature vector per physical WSI."""

    name = "slide_features"
    supported_feature_levels = frozenset({"slide"})
    output_representation_kind = "slide_vector"

    def run(
        self,
        bag: torch.Tensor | np.ndarray,
        sample: Any = None,
        **kwargs: Any,
    ) -> RetrievalRepresentation:
        """Return the original finite `(1, D)` matrix without transformation."""
        _ = kwargs
        raw_features = (
            bag.detach().cpu().numpy() if isinstance(bag, torch.Tensor) else np.asarray(bag)
        )
        if (
            not np.issubdtype(raw_features.dtype, np.number)
            or np.issubdtype(raw_features.dtype, np.bool_)
            or np.issubdtype(raw_features.dtype, np.complexfloating)
        ):
            raise ValueError("slide_features requires a real numeric feature matrix.")
        features = self.as_numpy_feature_matrix(raw_features)
        if features.shape[0] != 1 or features.shape[1] <= 0:
            raise ValueError(
                "slide_features requires one non-empty feature row with shape (1, D). "
                f"Got {features.shape}."
            )
        if not np.isfinite(features).all():
            raise ValueError("slide_features requires finite feature values.")
        return RetrievalRepresentation(
            sample_id=str(getattr(sample, "sample_id", "")),
            data=features,
            representation_type=self.output_representation_kind,
            feature_level="slide",
        )
