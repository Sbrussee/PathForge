from __future__ import annotations

from typing import Any, Literal

import torch

from pathbench.slide_retrieval.hyperparams import (
    HyperParam,
    collect_hyperparams,
    resolve_hyperparam,
)
from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)


FeatureLevel = Literal["patch", "slide"]
RepresentationKind = Literal["single_vector", "multi_vector"]


class BaseRetrievalRepresentationStrategy:
    """Base class for retrieval representation strategies."""

    name: str = ""

    # ---- compatibility specification ----
    supported_feature_levels: frozenset[FeatureLevel] = frozenset()
    output_representation_kind: RepresentationKind | str = ""

    def __init__(self, params: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self.params = params or {}
        self.extra = kwargs or {}
        self._bind_hyperparams()

    @classmethod
    def hyperparam_spec(cls) -> dict[str, dict[str, Any]]:
        """Return the hyperparameter schema for this strategy."""
        return {
            name: declaration.to_spec()
            for name, declaration in collect_hyperparams(cls).items()
        }

    def hyperparam_values(self) -> dict[str, Any]:
        """Return the effective hyperparameter values for this instance."""
        return {
            name: getattr(self, name)
            for name in collect_hyperparams(type(self))
        }

    def _get_hp(self, key: str) -> Any:
        """Resolve one hyperparameter value."""
        declarations = collect_hyperparams(type(self))
        if key not in declarations:
            raise KeyError(
                f"Hyperparam '{key}' is not declared on strategy '{type(self).__name__}'."
            )
        return resolve_hyperparam(
            name=key,
            declaration=declarations[key],
            params=self.params,
        )

    def _bind_hyperparams(self) -> None:
        """
        Resolve and bind all declared hyperparameters to instance attributes.

        Outputs:
        - `None`. Each declared hyperparameter becomes available on `self`
          using its declaration name.
        """
        for name in collect_hyperparams(type(self)):
            setattr(self, name, self._get_hp(name))

    def supports_feature_level(self, feature_level: str) -> bool:
        """Return whether this strategy supports the provided feature level."""
        return feature_level in self.supported_feature_levels

    def validate_feature_level(self, feature_level: str) -> None:
        """Validate that this strategy supports the provided feature level."""
        if not self.supported_feature_levels:
            raise ValueError(
                f"Strategy '{self.name}' does not define supported_feature_levels."
            )

        if feature_level not in self.supported_feature_levels:
            raise ValueError(
                f"Representation strategy '{self.name}' does not support "
                f"feature level '{feature_level}'. Supported levels: "
                f"{sorted(self.supported_feature_levels)}"
            )

    def run(
        self,
        bag: torch.Tensor,
        sample=None,
        **kwargs,
    ) -> RetrievalRepresentation:
        """Compute a retrieval representation from one bag."""
        raise NotImplementedError
