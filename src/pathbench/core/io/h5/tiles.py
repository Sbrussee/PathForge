# src/pathbench/core/io/h5/tiles.py
from __future__ import annotations

from typing import Any

import numpy as np

from pathbench.core.io.h5.base import (
    FileHandleH5,
    exists,
    read_array_dataset,
    read_json_dataset,
    write_array_dataset,
    write_json_dataset,
)
from pathbench.core.io.h5.layout import DEFAULT_LAYOUT, H5Layout


# ---- coords -----------------------------------------------------------------

def coords_exist(slide_artifact: FileHandleH5, bag_id: str, *, layout: H5Layout = DEFAULT_LAYOUT) -> bool:
    return exists(slide_artifact.h5, layout.coords_dataset(bag_id))


def read_coords(slide_artifact: FileHandleH5, bag_id: str, *, layout: H5Layout = DEFAULT_LAYOUT) -> np.ndarray:
    coords = read_array_dataset(slide_artifact.h5, layout.coords_dataset(bag_id))
    if coords.ndim != 2 or coords.shape[1] != 5:
        raise ValueError(f"coords must have shape (N,5). Got {coords.shape}.")
    return coords.astype(np.int32, copy=False)


def write_coords(slide_artifact: FileHandleH5, bag_id: str, coords: np.ndarray, *, layout: H5Layout = DEFAULT_LAYOUT) -> None:
    coords_array = np.asarray(coords)
    if coords_array.ndim != 2 or coords_array.shape[1] != 5:
        raise ValueError(f"coords must have shape (N,5). Got {coords_array.shape}.")
    write_array_dataset(slide_artifact.h5, layout.coords_dataset(bag_id), coords_array, dtype=np.int32)

def coords_num_rows(slide_artifact: FileHandleH5, bag_id: str, *, layout: H5Layout = DEFAULT_LAYOUT,) -> int:
    """
    Return number of coord rows for a bag without loading the entire dataset.
    If coords don't exist yet, returns 0.
    """
    if not coords_exist(slide_artifact, bag_id, layout=layout):
        return 0

    dset = slide_artifact.h5[layout.coords_dataset(bag_id)]
    shape = getattr(dset, "shape", None)
    if shape is None or len(shape) != 2 or int(shape[1]) != 5:
        raise ValueError(f"coords must have shape (N,5). Got {shape}.")
    return int(shape[0])

# ---- tiling_spec ------------------------------------------------------------

def tiling_spec_exists(slide_artifact: FileHandleH5, bag_id: str, *, layout: H5Layout = DEFAULT_LAYOUT) -> bool:
    return exists(slide_artifact.h5, layout.tiling_spec_dataset(bag_id))


def read_tiling_spec(
    slide_artifact: FileHandleH5,
    bag_id: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> dict[str, Any]:
    tiling_spec = read_json_dataset(slide_artifact.h5, layout.tiling_spec_dataset(bag_id))
    if not isinstance(tiling_spec, dict):
        raise ValueError("tiling_spec must be a JSON object (dict).")
    return tiling_spec


def write_tiling_spec(
    slide_artifact: FileHandleH5,
    bag_id: str,
    tiling_spec: dict[str, Any],
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
    if not isinstance(tiling_spec, dict):
        raise TypeError("tiling_spec must be a dict.")
    write_json_dataset(slide_artifact.h5, layout.tiling_spec_dataset(bag_id), tiling_spec)


def tiling_spec_matches(
    slide_artifact: FileHandleH5,
    bag_id: str,
    expected_tiling_spec: dict[str, Any],
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
    float_tol: float = 1e-6,
) -> bool:
    """
    Check whether stored tiling_spec matches the expected one for the provided keys.

    - Subset match: only keys in expected_tiling_spec are checked.
    - Numeric normalization:
        * tile_px / stride_px compared as ints
        * tile_mpp compared with tolerance
    """
    if not tiling_spec_exists(slide_artifact, bag_id, layout=layout):
        return False

    stored = read_tiling_spec(slide_artifact, bag_id, layout=layout)

    def _match_value(key: str, expected: Any, actual: Any) -> bool:
        if actual is None:
            return False

        # ints
        if key in {"tile_px", "stride_px"}:
            try:
                return int(actual) == int(expected)
            except Exception:
                return False

        # floats
        if key in {"tile_mpp"}:
            try:
                return abs(float(actual) - float(expected)) <= float_tol
            except Exception:
                return False

        # default exact match (strings, etc.)
        return actual == expected

    for k, v in expected_tiling_spec.items():
        if not _match_value(k, v, stored.get(k)):
            return False

    return True
