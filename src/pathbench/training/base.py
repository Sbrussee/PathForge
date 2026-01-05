from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch

from torch import nn

from pathbench.core.datasets.base import BagDatasetBase
from pathbench.core.models.mil_base import MILModelBase

@dataclass(frozen=True)
class TrainerOutput:
    """Result bundle for training runs."""
    best_model_path: str
    best_score: float





class TrainerBase(ABC):
    """
    Framework agnostic trainer base class.
    """
    
    @abstractmethod
    def fit(
        self,
        model: MILModelBase,
        dataset_train: BagDatasetBase,
        dataset_val: Optional[BagDatasetBase],
        loss_fn: nn.Module,
    ) -> TrainerOutput:
        """Train the model on the given dataset using the specified loss function."""
        raise NotImplementedError
    
    @abstractmethod
    def predict(
        self,
        model: MILModelBase,
        dataset: BagDatasetBase,
    ) -> torch.Tensor:
        """Generate predictions using the trained model on the given dataset."""
        raise NotImplementedError


    
@dataclass
class MILTrainer:
    """
    Convenience façade specifically for MIL use cases.

    This constrains the types to MILModelBase + BagDatasetBase, which
    matches the architecture diagram's “MILTrainer” box.
    """

    trainer: TrainerBase
    model: MILModelBase
    dataset_train: BagDatasetBase
    dataset_val: Optional[BagDatasetBase]
    loss_fn: nn.Module

    def run(self) -> TrainerOutput:
        return self.trainer.fit(
            model=self.model,
            dataset_train=self.dataset_train,
            dataset_val=self.dataset_val,
            loss_fn=self.loss_fn,
        )