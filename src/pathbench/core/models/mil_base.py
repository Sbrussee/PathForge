from __future__ import annotations
from abc import abstractmethod
from typing import Any, Dict, Optional, Union
from pathbench.core.models.base import TorchModelBase
import torch
import torch.nn as nn


class MILModelBase(TorchModelBase):
    """
    Base class for Deep MIL models.
    Expects input: (Batch, Bags, Dim).
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    @property
    @abstractmethod
    def bag_size(self) -> int | None:
        """Returns fixed bag size (int) or None for variable sizes."""
        ...

    @abstractmethod
    def forward_bag(
        self,
        bag: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
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

    def forward(
        self, bag: torch.Tensor, *args, **kwargs
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        return self.forward_bag(bag, *args, **kwargs)

    def instance_scores(
        self,
        bag: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return one instance score per bag element for heatmap inference.

        Args:
            bag: Bag feature tensor shaped ``[B, N, D]``.
            mask: Optional boolean padding mask shaped ``[B, N]``.
            coords: Optional coordinates shaped ``[B, N, 2]``.

        Returns:
            torch.Tensor: Instance score tensor shaped ``[B, N]``.

        Raises:
            AttributeError: If the model does not expose attention-like outputs
                that can be reduced to one score per instance.
            ValueError: If the returned attention tensor cannot be aligned to
                the bag's ``[B, N]`` instance axis.
        """

        try:
            output = self.forward_bag(
                bag,
                mask=mask,
                coords=coords,
                return_attention=True,
            )
        except TypeError:
            output = self.forward_bag(
                bag,
                mask=mask,
                coords=coords,
            )
        if not isinstance(output, dict) or "attention" not in output:
            raise AttributeError(
                f"{type(self).__name__} does not expose instance-level attention scores."
            )

        attention = output["attention"]
        if not isinstance(attention, torch.Tensor):
            raise TypeError("Model attention output must be a torch.Tensor.")
        if attention.ndim < 2:
            raise ValueError(
                f"Attention output must include batch and instance axes. Got {attention.shape}."
            )
        if attention.shape[:2] != bag.shape[:2]:
            raise ValueError(
                "Attention output must align with the bag batch and instance axes. "
                f"Got attention {attention.shape} for bag {bag.shape}."
            )

        scores = attention.float()
        while scores.ndim > 2:
            scores = scores.mean(dim=-1)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), 0.0)
        return scores


# Backward-compatible aliases still used by a few legacy model modules.
MILModel = MILModelBase
MILBase = MILModelBase
