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
        dataset: DatasetBase,
        task: Any,
    ) -> Any:
        """Train the model on the given dataset using the specified loss function and task."""
        pass
    
    @abstractmethod
    def predict(
        self,
        model: MILModelBase,
        dataset: DatasetBase,
        task: Any,
    ) -> Any:
        """Make predictions using the trained model on the given dataset."""
        pass
    
@dataclass
class MILTrainer:
    """
    Convenience façade specifically for MIL use cases.

    This constrains the types to MILModelBase + BagDatasetBase, which
    matches the architecture diagram's “MILTrainer” box.
    """

    trainer: TrainerBase
    model: MILModelBase
    dataset: BagDatasetBase
    task: Any
    loss: BaseLoss

    def run(self) -> None:
        self.trainer.fit(
            model=self.model,
            dataset=self.dataset,
            task=self.task,
            loss=self.loss,
        )
