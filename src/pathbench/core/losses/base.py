from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict
import torch
import torch.nn as nn

class BaseLoss(nn.Module, ABC):
    """
    Root abstract base class for all PathBench losses.
    """
    def __init__(self, task_type: str):
        super().__init__()
        self.task_type = task_type

    @abstractmethod
    def forward(self, preds: torch.Tensor, target: Any, **kwargs: Any) -> torch.Tensor:
        pass


# --- Intermediate Template Classes ---

class ClassificationLoss(BaseLoss):
    """
    Enforces standard classification inputs.
    """
    def __init__(self):
        super().__init__("classification")
        
    def forward(self, preds: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # Helper: Ensure target is Long for CE, Float for BCE? 
        # Usually CE expects Long, but we leave some flexibility or enforce based on shape.
        # For strictness, we can enforce preds is float.
        if not preds.is_floating_point():
            preds = preds.float()
        
        return self.calculate_loss(preds, target, **kwargs)

    @abstractmethod
    def calculate_loss(self, preds: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Implement the actual loss calculation."""
        pass


class RegressionLoss(BaseLoss):
    """
    Enforces inputs are Floats.
    """
    def __init__(self):
        super().__init__("regression")
        
    def forward(self, preds: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if not preds.is_floating_point():
            preds = preds.float()
        if not target.is_floating_point():
            target = target.float()
            
        return self.calculate_loss(preds, target, **kwargs)
    
    @abstractmethod
    def calculate_loss(self, preds: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        pass


class SurvivalContinuousLoss(BaseLoss):
    """
    Enforces target is a Dict with 'time' and 'event', and all are Floats.
    """
    def __init__(self):
        super().__init__("survival")
        
    def forward(self, preds: torch.Tensor, target: Dict[str, torch.Tensor], **kwargs: Any) -> torch.Tensor:
        if not isinstance(target, dict) or "time" not in target or "event" not in target:
             raise ValueError("SurvivalContinuousLoss expects target to be a dict with 'time' and 'event'.")

        # Enforce types and shapes
        preds = preds.float().reshape(-1)
        time = target["time"].float().reshape(-1)
        event = target["event"].float().reshape(-1)
        
        return self.calculate_loss(preds, time, event, **kwargs)

    @abstractmethod
    def calculate_loss(self, preds: torch.Tensor, time: torch.Tensor, event: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        pass


class SurvivalDiscreteLoss(BaseLoss):
    """
    Enforces target is a Dict with 'time' (Long/Int) and 'event' (Float/Binary).
    """
    def __init__(self):
        super().__init__("survival_discrete")
        
    def forward(self, preds: torch.Tensor, target: Dict[str, torch.Tensor], **kwargs: Any) -> torch.Tensor:
        if not isinstance(target, dict) or "time" not in target or "event" not in target:
             raise ValueError("SurvivalDiscreteLoss expects target to be a dict with 'time' and 'event'.")

        preds = preds.float()
        time = target["time"].long()
        event = target["event"].float()
        
        return self.calculate_loss(preds, time, event, **kwargs)

    @abstractmethod
    def calculate_loss(self, preds: torch.Tensor, time: torch.Tensor, event: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        pass