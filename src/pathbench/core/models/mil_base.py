from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional


class ModelBase(ABC):
    """
    Generic model abstraction (ML / DL / etc.).

    This is the root for all models in PathBench: classical ML models,
    CNNs, transformers, MIL models, etc.
    """
    
    input_dim: int
    output_dim: int
    
    # --- required methods ---
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Perform a forward pass. Signature is intentionally loose so that
        non-DL models or classical ML models can choose an appropriate API.
        """
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Common pattern to make models callable.
        return self.forward(*args, **kwargs)
    
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the model with optional configuration."""
        pass
    
    @abstractmethod
    def parameters(self) -> Iterable[Any]:
        """Return an iterable of the model's parameters."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to the specified path."""
        pass


class MILModelBase(ModelBase):
    """MIL-specific model."""

    @property
    @abstractmethod
    def bag_size(self) -> int | None:
        """None for variable-size bags, or an int for fixed-size."""
        ...

    @abstractmethod
    def forward_bag(self, bag: Any) -> Any:
        ...

    def forward(self, bag: Any, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return self.forward_bag(bag)
    
    @abstractmethod
    def return_attention_weights(self, bag: Iterable[Any]) -> Any:
        """Return attention weights for the instances in the bag."""
        pass
    
    @abstractmethod
    def extract_slide_embeddings(self, bag: Iterable[Any]) -> Any:
        """Extract slide-level embeddings from the bag of instances."""
        pass
    
    


    