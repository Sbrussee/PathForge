from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Mapping, Dict
from pathbench.core.annotations.base import AnnotationsBase
import pandas as pd

class CSVAnnotations(AnnotationsBase):
    def load_annotations(self, id: str) -> Dict:
        """Load annotations from a CSV file."""
        # Using pandas to read CSV, with automatic separator detection, and converting to dict.
        # Keys should be column names, values lists of column values.
        return pd.read_csv(id, sep=None, engine='python').to_dict(orient='list')
    
    def save_annotations(self, id: str, annotations: Dict):
        """Save annotations to a CSV file."""
        # Using pandas to save DataFrame to CSV
        df = pd.DataFrame(annotations)
        df.to_csv(id, index=False)
    
    def validate_annotations(self, annotations: Any) -> bool:
        """Validate the given annotations."""
        # At least needs a column 'slide', 
        pass
    