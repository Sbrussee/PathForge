# pathbench/utils/io_utils.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch

from pathbench.core.experiments.combinations import ComboConfig


def calculate_combinations(params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Calculate all combinations of the given parameters."""
    from itertools import product

    keys = params.keys()
    values = params.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    return combinations
