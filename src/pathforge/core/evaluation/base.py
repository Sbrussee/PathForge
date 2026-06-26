from __future__ import annotations

from abc import ABC, abstractmethod

from pathforge.core.evaluation.types import EvaluationRunContext
from pathforge.core.experiments.base import Experiment


class TaskEvaluationAdapterBase(ABC):
    """Base class for task-specific evaluation adapters."""

    task_name: str = ""

    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment
        self.cfg = experiment.cfg

    @classmethod
    @abstractmethod
    def get_discovery_keys(cls) -> list[str]:
        """Return the benchmark keys used to rediscover task runs."""
        raise NotImplementedError

    @abstractmethod
    def discover_runs(self) -> list[EvaluationRunContext]:
        """Discover all evaluable runs for the configured task."""
        raise NotImplementedError

    @abstractmethod
    def load_run_data(self, run_context: EvaluationRunContext) -> object:
        """Load and normalize one discovered run into metric-ready data."""
        raise NotImplementedError
