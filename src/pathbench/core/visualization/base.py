from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pathbench.core.experiments.base import Experiment
from pathbench.core.visualization.types import VisualizationRunContext


class TaskVisualizationAdapterBase(ABC):
    """Base class for task-specific visualization adapters."""

    task_name: str = ""

    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment
        self.cfg = experiment.cfg

    @abstractmethod
    def discover_runs(self) -> list[VisualizationRunContext]:
        """Discover all visualizable runs for the configured task."""
        raise NotImplementedError

    @abstractmethod
    def render_run(
        self,
        run_context: VisualizationRunContext,
        *,
        requested_visualizations: list[str],
        subset_ids: set[str] | None,
    ) -> list[Path]:
        """Render the requested visualizations for one discovered run."""
        raise NotImplementedError
