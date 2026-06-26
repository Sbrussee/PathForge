# =============================================================================
# Root Model Abstraction (Framework Agnostic)
# =============================================================================
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn


class ModelBase(ABC):
    """
    Root model abstraction for PathForge.

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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Universal inference entry point.
        - PyTorch: maps to __call__ -> forward()
        - Sklearn: maps to predict() or predict_proba()
        """
        raise NotImplementedError("Model subclasses must implement __call__.")

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


class ScikitBase(ModelBase):
    """
    Abstract base for scikit-learn / scikit-survival slide-level estimators.

    Concrete subclasses wrap a fitted sklearn estimator and expose a
    task-specific ``predict_as_tensor`` method so the shared
    ``save_task_evaluation_artifacts`` path can evaluate them without
    touching PyTorch training infrastructure.
    """

    @abstractmethod
    def fit(self, X: Any, y: Any) -> "ScikitBase":
        """Fit the estimator on numpy feature matrix X and targets y."""
        ...

    @abstractmethod
    def predict_as_tensor(self, X: Any) -> Any:
        """Return predictions as a torch.Tensor compatible with metrics helpers."""
        ...

    def save(self, path: str) -> None:
        """Persist the estimator to disk via pickle."""
        import pickle
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    def load(self, path: str) -> None:
        """Replace this instance's state from a pickle file."""
        import pickle
        with open(path, "rb") as fh:
            other = pickle.load(fh)
        self.__dict__.update(other.__dict__)


class TorchModelBase(ModelBase, nn.Module):
    """
    Canonical PyTorch implementation of the PathForge model interface.

    This centralizes the shared ``initialize``/``save``/``load``/parameter access
    behavior so MIL and slide-level model bases do not re-implement the same
    framework plumbing.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Preserve standard ``torch.nn.Module`` call semantics."""
        return nn.Module.__call__(self, *args, **kwargs)

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the model from an optional config dictionary.

        The default PyTorch implementation resets any submodule exposing a
        ``reset_parameters`` method. This keeps ``initialize()`` meaningful
        instead of leaving a silent no-op in the concrete torch-backed base.
        """

        _ = config
        for module in self.modules():
            if module is self:
                continue
            reset_parameters = getattr(module, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()

    def save(self, path: str) -> None:
        """Persist the model ``state_dict`` to disk."""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load the model ``state_dict`` from disk onto CPU memory."""
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def get_learnable_parameters(self) -> Iterable[torch.nn.Parameter]:
        """Yield all gradient-enabled parameters."""
        return (p for p in self.parameters() if p.requires_grad)
