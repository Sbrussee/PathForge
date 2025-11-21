# pathbench/core/datasets/bag_dataset.py
from dataclasses import dataclass
from typing import Any, Sequence
from pathench.core.datasets.base import BagDatasetBase
from pathbench.utils.registry import DATASETS
import torch

@dataclass
class BagDataset(BagDatasetBase):
    """
    A concrete implementation of BagDatasetBase.
    """
    _name: str = "BagDataset"
    _features: 


