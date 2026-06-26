from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ExplainerBase(ABC):
    """Base interface for prediction explainers.

    Explainability backends consume model-specific payloads and return one
    inspectable explanation artifact, such as a heatmap.
    """

    @abstractmethod
    def explain(self, input: Any) -> Any:
        """Build one explanation artifact from a backend-specific input payload."""

    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the explainer from runtime configuration."""
