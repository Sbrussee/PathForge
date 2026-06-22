# src/pathbench/core/io/h5/tiles.py
from __future__ import annotations

from typing import Any

import numpy as np

from pathbench.core.io.slide_artifacts.base import (
    FileHandleH5,
    get_dataset,
    is_complete,
    read_array_dataset,
    read_json_dataset,
    write_array_dataset,
    write_json_dataset,
)
from pathbench.core.io.slide_artifacts.layout import DEFAULT_LAYOUT, H5Layout


# ---- coords -----------------------------------------------------------------
def coords_exist(slide_artifact: FileHandleH5, bag_id: str, *, layout: H5Layout = DEFAULT_LAYOUT) -> bool:
    """Return whether a complete coordinate matrix exists for ``bag_id``."""
    path = layout.coords_dataset(bag_id)
    try:
        dset = get_dataset(slide_artifact.h5, path)
        if dset is None or not is_complete(dset):
            return False
        _validate_coords_shape(getattr(dset, "shape", None))
    except Exception:
        return False
    return True


def read_coords(slide_artifact: FileHandleH5, bag_id: str, *, layout: H5Layout = DEFAULT_LAYOUT) -> np.ndarray:
    """Read bag tile coordinates as an ``int32`` matrix shaped ``[N, 5]``."""
    path = layout.coords_dataset(bag_id)
    dset = get_dataset(slide_artifact.h5, path)
    if dset is None:
        raise KeyError(f"Missing coords dataset: {path}")
    if not is_complete(dset):
        raise ValueError(f"Coords dataset is incomplete: {path}")

    _validate_coords_shape(getattr(dset, "shape", None))

    coords = read_array_dataset(slide_artifact.h5, path)
    _validate_coords_shape(coords.shape)
    return coords.astype(np.int32, copy=False)


def write_coords(slide_artifact: FileHandleH5, bag_id: str, coords: np.ndarray, *, layout: H5Layout = DEFAULT_LAYOUT) -> None:
    """Write bag tile coordinates shaped ``[N, 5]`` as ``int32``."""
    coords_array = np.asarray(coords)
    if coords_array.ndim != 2 or coords_array.shape[1] != 5:
        raise ValueError(f"coords must have shape (N,5). Got {coords_array.shape}.")
    write_array_dataset(slide_artifact.h5, layout.coords_dataset(bag_id), coords_array, dtype=np.int32)

def coords_num_rows(slide_artifact: FileHandleH5, bag_id: str, *, layout: H5Layout = DEFAULT_LAYOUT,) -> int:
    """
    Return number of coord rows for a bag without loading the entire dataset.
    If coords don't exist yet, returns 0.
    """
    path = layout.coords_dataset(bag_id)
    dset = get_dataset(slide_artifact.h5, path)
    if dset is None or not is_complete(dset):
        return 0

    _validate_coords_shape(getattr(dset, "shape", None))
    shape = dset.shape
    return int(shape[0])

# ---- tiling_spec ------------------------------------------------------------


def tiling_spec_exists(slide_artifact: FileHandleH5, bag_id: str, *, layout: H5Layout = DEFAULT_LAYOUT) -> bool:
    """Return whether a complete tiling-spec JSON object exists for ``bag_id``."""
    path = layout.tiling_spec_dataset(bag_id)
    try:
        dset = get_dataset(slide_artifact.h5, path)
        if dset is None or not is_complete(dset):
            return False

        tiling_spec = read_json_dataset(slide_artifact.h5, path)
        _validate_tiling_spec_value(tiling_spec)
    except Exception:
        return False
    return True


def read_tiling_spec(
    slide_artifact: FileHandleH5,
    bag_id: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> dict[str, Any]:
    """Read the stored tiling-spec JSON object for one bag."""
    path = layout.tiling_spec_dataset(bag_id)
    dset = get_dataset(slide_artifact.h5, path)
    if dset is None:
        raise KeyError(f"Missing tiling_spec dataset: {path}")
    if not is_complete(dset):
        raise ValueError(f"Tiling spec dataset is incomplete: {path}")

    tiling_spec = read_json_dataset(slide_artifact.h5, path)
    _validate_tiling_spec_value(tiling_spec)
    return tiling_spec


def write_tiling_spec(
    slide_artifact: FileHandleH5,
    bag_id: str,
    tiling_spec: dict[str, Any],
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
    """Write the tiling-spec JSON object for one bag."""
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


# ---- tiles_overview ---------------------------------------------------------

def tiles_overview_exists(
    slide_artifact: FileHandleH5,
    bag_id: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> bool:
    """Return whether a complete serialized tiles-overview image exists for one bag."""
    path = layout.tiles_overview_dataset(bag_id)
    dset = get_dataset(slide_artifact.h5, path)
    if dset is None or not is_complete(dset):
        return False
    shape = getattr(dset, "shape", None)
    return bool(shape and len(shape) == 1)


def write_tiles_overview(
    slide_artifact: FileHandleH5,
    bag_id: str,
    image_bytes: bytes,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
    """
    Store compressed image bytes (e.g. JPEG) for the bag tiles overview.

    Stored as a 1D uint8 array for robust binary storage in HDF5.
    Existing dataset is overwritten by write_array_dataset.
    """
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        raise TypeError(f"image_bytes must be bytes-like. Got {type(image_bytes)}.")

    # memoryview avoids an unnecessary copy before np.frombuffer
    mv = memoryview(image_bytes)
    arr = np.frombuffer(mv, dtype=np.uint8)
    if arr.ndim != 1:
        raise ValueError("image_bytes must produce a 1D uint8 buffer.")

    write_array_dataset(
        slide_artifact.h5,
        layout.tiles_overview_dataset(bag_id),
        arr,
        dtype=np.uint8,
    )


def read_tiles_overview(
    slide_artifact: FileHandleH5,
    bag_id: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> bytes:
    """Read the serialized tiles-overview image bytes for one bag."""
    path = layout.tiles_overview_dataset(bag_id)
    dset = get_dataset(slide_artifact.h5, path)
    if dset is None:
        raise KeyError(f"Missing tiles_overview dataset: {path}")
    if not is_complete(dset):
        raise ValueError(f"tiles_overview dataset is incomplete: {path}")

    payload = read_array_dataset(slide_artifact.h5, layout.tiles_overview_dataset(bag_id))

    # Common/expected case: stored as 1D uint8 array
    if isinstance(payload, np.ndarray):
        arr = np.asarray(payload, dtype=np.uint8)
        if arr.ndim != 1:
            raise ValueError(f"tiles_overview must be a 1D uint8 array. Got shape {arr.shape}.")
        return arr.tobytes()

    # Robust fallback for scalar byte-like payloads (e.g. np.void / bytes / memoryview)
    try:
        return bytes(payload)
    except Exception as e:
        raise ValueError(
            f"tiles_overview payload could not be converted to bytes (type={type(payload)!r})."
        ) from e


def _validate_coords_shape(shape: tuple[int, ...] | None) -> None:
    if shape is None or len(shape) != 2 or int(shape[1]) != 5:
        raise ValueError(f"coords must have shape (N,5). Got {shape}.")


def _validate_tiling_spec_value(tiling_spec: Any) -> None:
    if not isinstance(tiling_spec, dict):
        raise ValueError("tiling_spec must be a JSON object (dict).")
