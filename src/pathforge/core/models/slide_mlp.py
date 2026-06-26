from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from pathforge.core.models.mil_base import MILModelBase
from pathforge.core.models.slide_base import SlideLevelModel
from pathforge.core.registry import MODELS


@MODELS.register("SlideVectorMLP")
class SlideVectorMLP(SlideLevelModel, MILModelBase):
    """MLP applied to mean-pooled slide-level feature vectors.

    Inherits from both :class:`SlideLevelModel` and :class:`MILModelBase` so
    it can be trained through the standard :class:`LightningTrainer` pipeline.
    ``forward_bag`` mean-pools the bag ``(B, N, D)`` → ``(B, D)`` before
    applying the MLP, making it usable on any bag regardless of bag size.

    Args:
        input_dim: Feature dimension per instance.
        hidden_dim: Hidden layer width.
        output_dim: Number of output logits (classes / time bins / 1 for
            regression and continuous survival).
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    @property
    def bag_size(self) -> None:
        return None

    def forward_bag(
        self,
        bag: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        loss_fn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """Mean-pool the bag then forward through the MLP.

        Args:
            bag: ``(B, N, D)`` feature bag.
            mask: Ignored (variable-length bags are mean-pooled anyway).
            coords: Ignored.
            label: Optional target for internal loss computation.
            loss_fn: Optional loss module for internal loss computation.

        Returns:
            Logits tensor ``(B, output_dim)`` or dict with ``logits`` and
            ``loss`` when both ``label`` and ``loss_fn`` are provided.
        """
        x = bag.float().mean(dim=1)
        return self.forward_slide(x, label=label, loss_fn=loss_fn)

    def forward_slide(
        self,
        x: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        logits = self.mlp(x)
        if loss_fn is not None and label is not None:
            return {"logits": logits, "loss": loss_fn(logits, label)}
        return logits
