from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn

from pathbench.core.losses.base import ClassificationLoss
from pathbench.utils.registries import LOSSES

@LOSSES.register("CrossEntropyLoss")
class CrossEntropyLoss(ClassificationLoss):
    def __init__(self, weight: torch.Tensor | None = None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight)
        
    def calculate_loss(self, preds: torch.Tensor, target: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.loss(preds, target)