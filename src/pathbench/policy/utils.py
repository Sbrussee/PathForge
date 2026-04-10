# pathbench/utils/io_utils.py

from __future__ import annotations

from typing import Any, Dict, List




def calculate_combinations(params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Calculate all combinations of the given parameters."""
    from itertools import product

    keys = params.keys()
    values = params.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    return combinations
