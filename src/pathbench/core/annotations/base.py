# src/pathbench/core/annotations/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class AnnotationsBase(ABC):
    """
    Abstraction over any annotation storage backend.
    """

    @abstractmethod
    def load_annotations(self, id: str) -> Any:
        """Load annotations from a given identifier."""
        pass
    
    @abstractmethod
    def save_annotations(self, id: str, annotations: Any) -> None:
        """Save annotations to a given identifier."""
        pass
    
    @abstractmethod
    def validate_annotations(self, annotations: Any) -> bool:
        """Validate the given annotations."""
        pass
    
    @abstractmethod
    def inspect_annotations(self, annotations: Any) -> None:
        """Print a summary of the annotations."""
        pass