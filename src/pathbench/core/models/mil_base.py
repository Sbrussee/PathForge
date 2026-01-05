from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Iterable, Union, Sequence
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
    #TODO: Use appropriate weight initialization strategies (Xavier, Kaiming, etc.) based on layer types.
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize model weights.

        If config includes a ``weights_path`` key, load those weights.
        Otherwise, reset parameters for all submodules that implement
        ``reset_parameters`` (common in torch.nn layers).
        """
        if config and config.get("weights_path"):
            self.load(config["weights_path"])
            return

        for module in self.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location='cpu'))

    def get_learnable_parameters(self) -> Iterable[torch.nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)