from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


from pathbench.core.datasets.base import BagDatasetBase, DatasetBase
from pathbench.core.losses.base import BaseLoss
from pathbench.core.models.mil_base import MILModelBase


class TrainerBase(ABC):
    """
    Framework agnostic trainer base class.
    """
    
    @abstractmethod
    def fit(
        self,
        model: MILModelBase,
        dataset_train: DatasetBase,
        dataset_val: DatasetBase,
        loss_func: BaseLoss,
    ) -> Any:
        """Train a model on train/validation datasets and return trainer-specific results."""
        ...
    
    @abstractmethod
    def predict(
        self,
        model: MILModelBase,
        dataset: DatasetBase,
    ) -> Any:
        """Run inference on a dataset using a trained model."""
        ...
    
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
    dataset_val: BagDatasetBase
    loss: BaseLoss

    def run(self) -> Any:
        return self.trainer.fit(
            model=self.model,
            dataset_train=self.dataset_train,
            dataset_val=self.dataset_val,
            loss_func=self.loss,
        )
