from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
#from pathbench.utils.constants import POLICY_MODES

from pathbench.core.experiments.base import Experiment
    
class PolicyBase(ABC):
    """Base class for policy modes in pathbench."""
    
    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment
        self.cfg = experiment.cfg
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the policy mode in the given experiment."""
        pass