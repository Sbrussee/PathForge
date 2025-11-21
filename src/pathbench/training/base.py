from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


from pathbench.core.datasets.base import DatasetBase
from pathbench.core.losses.base import LossBase
from pathbench.core.models.base import MILBase
from pathbench.core.tasks.base import TaskBase


class TrainerBase(ABC):
    """
    Framework agnostic trainer base class.
    """
    
    @abstractmethod
    def fit(
        self,
        model: MILBase,
        dataset: DatasetBase,
        task: TaskBase,
    ) -> Any:
        """Train the model on the given dataset using the specified loss function and task."""
        pass
    
    @abstractmethod
    def predict(
        self,
        model: MILBase,
        dataset: DatasetBase,
        task: TaskBase,
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
    task: TaskBase
    loss: LossBase

    def run(self) -> None:
        self.trainer.fit(
            model=self.model,
            dataset=self.dataset,
            task=self.task,
            loss=self.loss,
            policy=self.policy,
        )