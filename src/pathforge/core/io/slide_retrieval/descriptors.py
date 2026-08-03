from __future__ import annotations

from typing import Any

import numpy as np

from pathforge.core.io.slide_artifacts.base import (
    FileHandleH5,
    exists,
    read_array_dataset,
    write_array_dataset,
)
from pathforge.core.io.slide_retrieval.layout import (
    DEFAULT_LAYOUT,
    RetrievalH5Layout,
)


# ---------------------------------------------------------------------------
# Descriptor API
# ---------------------------------------------------------------------------


def descriptor_exists(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    descriptor_name: str,
    *,
    expected_rows: int | None = None,
    expected_dim: int | None = None,
    expected_metadata: dict[str, Any] | None = None,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> bool:
    """Return whether one descriptor matches its expected cache contract.

    Inputs:
        expected_rows: Optional expected row count for the ``(N, D)`` matrix.
        expected_dim: Optional expected descriptor width.
        expected_metadata: Optional HDF5 attribute values required for reuse.

    Output:
        ``True`` only when the dataset exists and every supplied constraint
        matches.

    Example:
        ``descriptor_exists(h5, "256px_0.5mpp", "mean_rgb", expected_rows=20)``.
    """
    path = layout.descriptor(tile_id, descriptor_name)
    if not exists(retrieval_artifact.h5, path):
        return False
    dset = retrieval_artifact.h5[path]

    try:
        _validate_descriptor_shape(
            getattr(dset, "shape", None),
            expected_rows=expected_rows,
            expected_dim=expected_dim,
        )
        return _descriptor_metadata_matches(dset.attrs, expected_metadata)
    except Exception:
        return False


def read_descriptor(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    descriptor_name: str,
    *,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> np.ndarray:
    """Read one retrieval-side descriptor matrix."""
    path = layout.descriptor(tile_id, descriptor_name)
    if not exists(retrieval_artifact.h5, path):
        raise KeyError(f"Missing descriptor: {path}")

    descriptor_matrix = read_array_dataset(
        retrieval_artifact.h5,
        path,
    )
    _validate_descriptor_shape(descriptor_matrix.shape, descriptor_name=descriptor_name)
    return descriptor_matrix.astype(np.float32, copy=False)


def write_descriptor(
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    descriptor_name: str,
    descriptor_matrix: np.ndarray,
    *,
    metadata: dict[str, Any] | None = None,
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> None:
    """Write one retrieval-side descriptor matrix."""
    descriptor_array = np.asarray(descriptor_matrix)
    _validate_descriptor_shape(descriptor_array.shape, descriptor_name=descriptor_name)
    path = layout.descriptor(tile_id, descriptor_name)
    write_array_dataset(
        retrieval_artifact.h5,
        path,
        descriptor_array,
        dtype=np.float32,
    )
    for key, value in dict(metadata or {}).items():
        retrieval_artifact.h5[path].attrs[str(key)] = value


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_descriptor_shape(
    shape: tuple[int, ...] | None,
    *,
    descriptor_name: str = "descriptor",
    expected_rows: int | None = None,
    expected_dim: int | None = None,
) -> None:
    if not shape or len(shape) != 2:
        raise ValueError(f"{descriptor_name!r} must have shape (N,D). Got {shape}.")
    if expected_rows is not None and int(shape[0]) != int(expected_rows):
        raise ValueError(
            f"{descriptor_name!r} rows must match expected rows: "
            f"expected {expected_rows}, got {shape[0]}."
        )
    if expected_dim is not None and int(shape[1]) != int(expected_dim):
        raise ValueError(
            f"{descriptor_name!r} dim must match expected dim: "
            f"expected {expected_dim}, got {shape[1]}."
        )


def _descriptor_metadata_matches(
    attrs: Any,
    expected_metadata: dict[str, Any] | None,
) -> bool:
    """Return whether HDF5 attributes match one expected metadata mapping.

    Inputs:
        attrs: HDF5 dataset attributes.
        expected_metadata: Attribute names and values required for reuse.

    Output:
        ``True`` when every expected attribute matches, otherwise ``False``.

    Example:
        ``_descriptor_metadata_matches(dataset.attrs, {"crop_px": 1024})``.
    """
    if not expected_metadata:
        return True
    for key, expected in expected_metadata.items():
        actual = attrs.get(str(key))
        if isinstance(actual, bytes):
            actual = actual.decode("utf-8")
        if isinstance(expected, float):
            try:
                if abs(float(actual) - expected) > 1e-9:
                    return False
            except (TypeError, ValueError):
                return False
        elif actual != expected:
            return False
    return True
