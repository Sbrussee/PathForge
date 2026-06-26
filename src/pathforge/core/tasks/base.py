from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pathforge.config.config import Config
from pathforge.core.experiments.base import Experiment
from pathforge.core.experiments.combinations import ComboConfig
from pathforge.core.datasets.bag_dataset import BagDataset


class TaskBase(ABC):
    """
    Base class for benchmarking tasks.

    Each concrete task should define:
    - which benchmark grid keys it needs
    - how one combo is executed
    """

    grid_keys: list[str] = []
    allowed_dataset_uses: frozenset[str] | None = None
    inference_dataset_uses: frozenset[str] | None = None
    inference_input_use: str = "query"

    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment
        self.cfg: Config = experiment.cfg

    @classmethod
    def get_grid_keys(cls) -> list[str]:
        return cls.grid_keys

    @classmethod
    def get_allowed_dataset_uses(cls) -> frozenset[str] | None:
        return cls.allowed_dataset_uses

    @classmethod
    def get_inference_grid_keys(cls) -> list[str]:
        return cls.get_grid_keys()

    @classmethod
    def get_inference_dataset_uses(cls) -> frozenset[str] | None:
        return cls.inference_dataset_uses

    @classmethod
    def get_inference_input_use(cls) -> str:
        return cls.inference_input_use

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

    def inference(
        self,
        combo_cfg: ComboConfig,
        datasets_by_use: dict[str, list[BagDataset]],
        inference_run_root: Path,
    ) -> dict[str, Any]:
        """
        Execute one inference run for one combo.

        Args:
            combo_cfg: Active inference combination.
            datasets_by_use: dictionary of datasets grouped by task-specific use.
            inference_run_root: timestamped root folder for the current
                inference CLI invocation.

        Returns:
            Dictionary with output paths / counts / metadata.
        """
        raise NotImplementedError(
            f"Task '{type(self).__name__}' does not implement inference."
        )
