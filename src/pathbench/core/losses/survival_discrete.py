from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathbench.core.losses.base import SurvivalDiscreteLoss
from pathbench.utils.registries import LOSSES

try:
    from pycox.models.loss import NLLLogisticHazard as _PyCoxLogHazNLL
except ImportError:
    _PyCoxLogHazNLL = None


@LOSSES.register("DiscreteTimeNLLLoss")
class DiscreteTimeNLLLoss(SurvivalDiscreteLoss):
    def __init__(self):
        super().__init__()
        self.impl = _PyCoxLogHazNLL() if _PyCoxLogHazNLL is not None else None

        if self.impl is None:
            self.bcel = nn.BCEWithLogitsLoss(reduction="none")

    def calculate_loss(
        self, preds: torch.Tensor, time: torch.Tensor, event: torch.Tensor, **_: Any
    ) -> torch.Tensor:
        if self.impl is not None:
            return self.impl(preds, time, event)

        if preds.ndim != 2:
            raise ValueError(
                f"DiscreteTimeNLLLoss expects preds with shape [N, T], got {tuple(preds.shape)}."
            )

        n_samples, n_bins = preds.shape
        if time.ndim != 1 or event.ndim != 1:
            raise ValueError("Discrete survival targets must be rank-1 tensors.")
        if len(time) != n_samples or len(event) != n_samples:
            raise ValueError(
                "Predictions and targets must have the same batch dimension."
            )
        if torch.any(time < 0) or torch.any(time >= n_bins):
            raise ValueError(
                "Discrete survival target times must fall within the hazard bins."
            )

        device = preds.device
        bin_index = torch.arange(n_bins, device=device).unsqueeze(0)
        time = time.unsqueeze(1)

        log_hazard = F.logsigmoid(preds)
        log_survival = F.logsigmoid(-preds)

        before_event = (bin_index < time).float()
        at_event = (bin_index == time).float()
        through_censor = (bin_index <= time).float()

        event_loss = -(
            (before_event * log_survival).sum(dim=1)
            + (at_event * log_hazard).sum(dim=1)
        )
        censor_loss = -(through_censor * log_survival).sum(dim=1)

        loss = event * event_loss + (1.0 - event) * censor_loss
        return loss.mean()
