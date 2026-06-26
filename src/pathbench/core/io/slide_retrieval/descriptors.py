from __future__ import annotations

import numpy as np

from pathbench.core.io.slide_artifacts.base import (
    FileHandleH5,
    exists,
    read_array_dataset,
    write_array_dataset,
)
from pathbench.core.io.slide_retrieval.layout import (
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
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> bool:
    """Return whether one descriptor exists and matches the optional shape."""
    path = layout.descriptor(tile_id, descriptor_name)
    if not exists(retrieval_artifact.h5, path):
        return False

    if expected_rows is None and expected_dim is None:
        return True

    dset = retrieval_artifact.h5[path]
    try:
        _validate_descriptor_shape(
            getattr(dset, "shape", None),
            expected_rows=expected_rows,
            expected_dim=expected_dim,
        )
        return True
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
    layout: RetrievalH5Layout = DEFAULT_LAYOUT,
) -> None:
    """Write one retrieval-side descriptor matrix."""
    descriptor_array = np.asarray(descriptor_matrix)
    _validate_descriptor_shape(descriptor_array.shape, descriptor_name=descriptor_name)
    write_array_dataset(
        retrieval_artifact.h5,
        layout.descriptor(tile_id, descriptor_name),
        descriptor_array,
        dtype=np.float32,
    )


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
        raise ValueError(
            f"{descriptor_name!r} must have shape (N,D). Got {shape}."
        )
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
