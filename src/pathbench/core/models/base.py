# =============================================================================
# Root Model Abstraction (Framework Agnostic)
# =============================================================================
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Iterable

class ModelBase(ABC):
    """
    Root model abstraction for PathBench.
    
    This class is framework-agnostic. Implementations could be:
    - PyTorch models (nn.Module)
    - Scikit-learn estimators
    - XGBoost/LightGBM boosters
    """
    
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the model.
        For PyTorch: Load weights, reset parameters.
        For Sklearn: Configure hyperparameters.
        """
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Universal inference entry point.
        - PyTorch: maps to __call__ -> forward()
        - Sklearn: maps to predict() or predict_proba()
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the model to disk."""
        pass
        
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model from disk."""
        pass
        
    def get_learnable_parameters(self) -> Iterable[Any]:
        """
        Return parameters for optimization.
        Returns empty iterator for non-gradient models (e.g. Random Forest).
        """
        return []