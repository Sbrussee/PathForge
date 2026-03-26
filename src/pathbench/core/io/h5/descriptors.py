from __future__ import annotations

import numpy as np

from pathbench.core.io.h5.base import (
    FileHandleH5,
    exists,
    read_array_dataset,
    write_array_dataset,
)
from pathbench.core.io.h5.layout import DEFAULT_LAYOUT, H5Layout


def descriptor_exists(
    slide_artifact: FileHandleH5,
    bag_id: str,
    descriptor_name: str,
    *,
    expected_rows: int | None = None,
    expected_dim: int | None = None,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> bool:
    """
    Check whether one row-aligned patch descriptor matrix exists in H5.

    Inputs:
    - `slide_artifact`: open H5 file handle.
    - `bag_id`: canonical tiling identifier.
    - `descriptor_name`: descriptor dataset name under `descriptors/`.
    - `expected_rows`: optional expected row count.
    - `expected_dim`: optional expected descriptor width.

    Returns:
    - `bool`: `True` when the dataset exists and matches the optional shape.

    Example:
    ```python
    descriptor_exists(
        slide_artifact,
        bag_id="256px_0.5mpp",
        descriptor_name="mean_rgb",
        expected_rows=1024,
        expected_dim=3,
    )
    ```
    """
    path = layout.descriptor_dataset(bag_id, descriptor_name)
    if not exists(slide_artifact.h5, path):
        return False

    if expected_rows is None and expected_dim is None:
        return True

    dset = slide_artifact.h5[path]
    shape = getattr(dset, "shape", None)
    if not shape or len(shape) != 2:
        return False
    if expected_rows is not None and int(shape[0]) != int(expected_rows):
        return False
    if expected_dim is not None and int(shape[1]) != int(expected_dim):
        return False
    return True


def read_descriptor(
    slide_artifact: FileHandleH5,
    bag_id: str,
    descriptor_name: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> np.ndarray:
    """
    Load one row-aligned patch descriptor matrix from H5.

    Inputs:
    - `slide_artifact`: open H5 file handle.
    - `bag_id`: canonical tiling identifier.
    - `descriptor_name`: descriptor dataset name under `descriptors/`.

    Returns:
    - `np.ndarray[float32]` with shape `(N, D)`.
    """
    descriptor_matrix = read_array_dataset(
        slide_artifact.h5,
        layout.descriptor_dataset(bag_id, descriptor_name),
    )
    if descriptor_matrix.ndim != 2:
        raise ValueError(
            f"descriptor '{descriptor_name}' must have shape (N,D). "
            f"Got {descriptor_matrix.shape}."
        )
    return descriptor_matrix.astype(np.float32, copy=False)


def write_descriptor(
    slide_artifact: FileHandleH5,
    bag_id: str,
    descriptor_name: str,
    descriptor_matrix: np.ndarray,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
    """
    Persist one row-aligned patch descriptor matrix in H5.

    Inputs:
    - `slide_artifact`: open H5 file handle.
    - `bag_id`: canonical tiling identifier.
    - `descriptor_name`: descriptor dataset name under `descriptors/`.
    - `descriptor_matrix`: array with shape `(N, D)`.

    Returns:
    - `None`.
    """
    descriptor_array = np.asarray(descriptor_matrix)
    if descriptor_array.ndim != 2:
        raise ValueError(
            f"descriptor_matrix for '{descriptor_name}' must have shape (N,D). "
            f"Got {descriptor_array.shape}."
        )
    write_array_dataset(
        slide_artifact.h5,
        layout.descriptor_dataset(bag_id, descriptor_name),
        descriptor_array,
        dtype=np.float32,
    )
