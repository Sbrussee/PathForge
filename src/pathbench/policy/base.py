from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathbench.utils.constants import POLICY_MODES

class PolicyBase(ABC):
    """Base class for policy modes in pathbench."""
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the policy mode."""
        pass