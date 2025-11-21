from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn

from pathbench.core.losses.base import RegressionLoss
from pathbench.utils.registries import LOSSES

@LOSSES.register("MSELoss")
class MSELoss(RegressionLoss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
        
    def calculate_loss(self, preds: torch.Tensor, target: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.loss(preds, target)