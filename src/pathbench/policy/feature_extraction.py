"""
This code should implement feature extraction, so will only need to define
a FeatureExtractionPolicy class that inherits from PolicyBase and implements
the necessary methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from pathbench.policy.base import PolicyBase

class FeatureExtractionPolicy(PolicyBase):
    """Policy for feature extraction."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
    
    @abstractmethod
    def extract_features(self, data: Any) -> Any:
        """Extract features from the given data."""
        pass
    
    def execute(self, data: Any) -> Any:
        """Execute the feature extraction process."""
        return self.extract_features(data)