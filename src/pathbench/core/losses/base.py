from __future__ import annotations
import torch
from torch import nn

class Loss(nn.Module):
    def forward(self, preds: torch.Tensor, target) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class SurvivalLoss(Loss):
    def forward(self, preds, target) -> torch.Tensor:
        # placeholder: implement neg log-likelihood for Cox
        return preds.sum() * 0.0

class ClassificationLoss(Loss):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
    def forward(self, preds, target) -> torch.Tensor:
        return self.loss(preds, target)

class RegressionLoss(Loss):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
    def forward(self, preds, target) -> torch.Tensor:
        return self.loss(preds, target)