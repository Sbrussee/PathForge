from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Tuple
import torch
import pandas as pd
from pathlib import Path

from pathbench.core.datasets.base import BagDatasetBase

@dataclass
class BagDataset(BagDatasetBase):
    """
    Concrete implementation of a MIL Bag Dataset.
    Assumes features are pre-extracted and stored in .pt (torch) files.
    """
    _name: str
    feature_path: str
    annotation_path: str
    target_column: str
    
    def __post_init__(self):
        self.annotations = pd.read_csv(self.annotation_path)
        # Validate paths
        if not Path(self.feature_path).exists():
             raise FileNotFoundError(f"Feature path {self.feature_path} does not exist.")

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_bags(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        """
        Returns:
            bag: (N, D) Tensor
            label: scalar
        """
        row = self.annotations.iloc[index]
        slide_id = row['slide_id']
        label = row[self.target_column]
        
        # Construct path to feature file (assuming {slide_id}.pt pattern)
        # In production, consider a more robust path mapping strategy
        f_path = Path(self.feature_path) / f"{slide_id}.pt"
        
        if not f_path.exists():
            # Handle missing files gracefully or raise
            raise FileNotFoundError(f"Features for slide {slide_id} not found at {f_path}")
            
        bag = torch.load(f_path)
        
        # Ensure bag is float32 and correct shape
        if isinstance(bag, torch.Tensor):
            bag = bag.float()
        
        return bag, label