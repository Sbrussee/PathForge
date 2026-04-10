from __future__ import annotations
from abc import abstractmethod
from typing import Any, Dict, Optional, Iterable, Union
from pathbench.core.models.base import ModelBase
import torch
import torch.nn as nn


class MILModelBase(ModelBase, nn.Module):
    """
    Base class for Deep MIL models.
    Expects input: (Batch, Bags, Dim).
    """
    
    def __init__(self, **kwargs):
        super().__init__()

    @property
    @abstractmethod
    def bag_size(self) -> int | None:
        """Returns fixed bag size (int) or None for variable sizes."""
        ...

    @abstractmethod
    def forward_bag(self, 
                    bag: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None, 
                    coords: Optional[torch.Tensor] = None,
                    label: Optional[torch.Tensor] = None,
                    loss_fn: Optional[nn.Module] = None) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Core MIL logic.
        
        Args:
            bag: (B, N, D) features.
            mask: (B, N) mask.
            coords: (B, N, 2) spatial coordinates.
            label: (B,) Ground truth labels (optional, for internal loss calc).
            loss_fn: Loss function module (optional, for internal loss calc).
            
        Returns:
            logits (Tensor) OR Dict containing 'logits' and 'loss'.
        """
        ...

    def forward(self, bag: torch.Tensor, *args, **kwargs) -> Union[torch.Tensor, Dict[str, Any]]:
        return self.forward_bag(bag, *args, **kwargs)

    # --- PyTorch Implementation of ModelBase ---
    #TODO: PUT ACTUAL INTIALIZE LOGIC IN HERE
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location='cpu'))

    def get_learnable_parameters(self) -> Iterable[torch.nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)