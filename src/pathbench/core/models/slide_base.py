from __future__ import annotations
from abc import abstractmethod
from typing import Any, Optional, Dict, Iterable, Union
import torch
import torch.nn as nn
from pathbench.core.models.base import TorchModelBase

# =============================================================================
# Slide-Level Model Abstraction (Vector Input)
# =============================================================================
class SlideLevelModel(TorchModelBase):
    """
    Base class for models that operate on pre-aggregated slide vectors.
    Expects input: (Batch, Dim).
    """
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward_slide(self, 
                      x: torch.Tensor, 
                      label: Optional[torch.Tensor] = None, 
                      loss_fn: Optional[nn.Module] = None) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Core logic for vector-based models.
        Args:
            x: (B, Input_Dim) feature vector.
        """
        ...

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Union[torch.Tensor, Dict[str, Any]]:
        return self.forward_slide(x, *args, **kwargs)
