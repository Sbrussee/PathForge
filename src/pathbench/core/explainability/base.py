# src/pathbench/core/explainability/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class ExplainerBase(ABC):
    """Explain predictions, e.g. via heatmaps or attention maps."""

    @abstractmethod
    def explain(self, input: Any) -> Any:
        ...
        
    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        ...
        
    
    