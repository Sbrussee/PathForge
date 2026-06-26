from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol


class ExperimentLike(Protocol):
    """Minimal structural contract required by policy implementations."""

    cfg: Any
    project_root: str | None


class PolicyBase(ABC):
    """
    Base class for PathForge policies.

    Policies may be constructed either from an experiment-like object exposing a
    ``cfg`` attribute or directly from a validated config object in legacy code
    paths. In both cases ``self.cfg`` is guaranteed to point at the active
    configuration object.
    """

    def __init__(self, experiment: ExperimentLike | Any) -> None:
        self.experiment = experiment
        self.cfg = experiment.cfg if hasattr(experiment, "cfg") else experiment

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the policy mode in the given experiment context."""
        ...
