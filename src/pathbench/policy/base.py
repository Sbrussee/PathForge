from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
#from pathbench.utils.constants import POLICY_MODES


class ExperimentLike(Protocol):
    cfg: Any
    project_root: str | None
    
class PolicyBase(ABC):
    """Base class for policy modes in pathbench."""
    
    def __init__(self, experiment: ExperimentLike) -> None:
        self.experiment = experiment
        self.cfg = experiment.cfg
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the policy mode in the given experiment."""
        pass