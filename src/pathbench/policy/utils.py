# pathbench/utils/io_utils.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch

def calculate_combinations(params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Calculate all combinations of the given parameters."""
    from itertools import product

    keys = params.keys()
    values = params.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    return combinations

class ComboConfig:
    """
    Generic, dynamically-populated combo.

    Any key you pass in becomes an attribute:
        Combo(feature_extraction="virchow", tile_px=256)
        -> combo.feature_extraction, combo.tile_px
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_keys_values(cls, keys: list[str], values: list[object]) -> "ComboConfig":
        data = {k: v for k, v in zip(keys, values)}
        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)  
