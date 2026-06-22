from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pathbench.config.config import Config
from pathbench.core.experiments.base import Experiment
from pathbench.core.experiments.combinations import ComboConfig
from pathbench.core.datasets.bag_dataset import BagDataset


class TaskBase(ABC):
    """
    Base class for benchmarking tasks.

    Each concrete task should define:
    - which benchmark grid keys it needs
    - how one combo is executed
    """

    grid_keys: list[str] = []
    allowed_dataset_uses: frozenset[str] | None = None

    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment
        self.cfg: Config = experiment.cfg

    @classmethod
    def get_grid_keys(cls) -> list[str]:
        return cls.grid_keys

    @classmethod
    def get_allowed_dataset_uses(cls) -> frozenset[str] | None:
        return cls.allowed_dataset_uses

    @abstractmethod
    def execute(
        self,
        combo_cfg: ComboConfig,
        datasets_by_use: dict[str, list[BagDataset]],
    ) -> dict[str, Any]:
        """
        Execute one benchmark run for one combo.

        Args:
            combo_cfg: Active benchmark combination.
            datasets_by_use: dictionary of datasets grouped by their use case.

        Returns:
            Dictionary with results / metrics / metadata.
        """
        raise NotImplementedError
