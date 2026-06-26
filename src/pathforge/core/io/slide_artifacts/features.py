# src/pathforge/core/io/h5/features.py
from __future__ import annotations

import numpy as np

from pathforge.core.io.slide_artifacts.base import (
    FileHandleH5,
    get_dataset,
    is_complete,
    read_array_dataset,
    write_array_dataset,
)
from pathforge.core.io.slide_artifacts.layout import DEFAULT_LAYOUT, H5Layout
from pathforge.core.io.slide_artifacts import tiles as tiles_io

def features_exist(
    slide_artifact: FileHandleH5,
    bag_id: str,
    extractor_name: str,
    *,
    expected_rows: int | None = None,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> bool:
    """Return whether a complete feature matrix exists for one bag/extractor pair."""
    path = layout.features_dataset(bag_id, extractor_name)
    try:
        dset = get_dataset(slide_artifact.h5, path)
        if dset is None or not is_complete(dset):
            return False

        _validate_features_shape(
            getattr(dset, "shape", None),
            expected_rows=expected_rows,
        )
    except Exception:
        return False
    return True


def read_features(
    slide_artifact: FileHandleH5,
    bag_id: str,
    extractor_name: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> np.ndarray:
    """Read a feature matrix shaped ``[N, D]`` as ``float32``."""
    path = layout.features_dataset(bag_id, extractor_name)
    dset = get_dataset(slide_artifact.h5, path)
    if dset is None:
        raise KeyError(f"Missing features dataset: {path}")
    if not is_complete(dset):
        raise ValueError(f"Features dataset is incomplete: {path}")

    _validate_features_shape(getattr(dset, "shape", None))

    feature_matrix = read_array_dataset(slide_artifact.h5, path)
    _validate_features_shape(feature_matrix.shape)
    return feature_matrix.astype(np.float32, copy=False)


def infer_feature_level(
    slide_artifact: FileHandleH5,
    bag_id: str,
    extractor_name: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> str:
    """
    Infer feature level for one slide artifact.

    Returns:
        - "patch"
        - "slide"
        - "unknown"
        - "invalid"
    """
    try:
        feature_matrix = read_features(
            slide_artifact,
            bag_id,
            extractor_name,
            layout=layout,
        )
        n_features = int(feature_matrix.shape[0])
        n_patches = tiles_io.coords_num_rows(
            slide_artifact,
            bag_id=bag_id,
            layout=layout,
        )
    except Exception:
        return "invalid"

    if n_features <= 0 or n_patches <= 0:
        return "invalid"

    if n_features == n_patches:
        if n_patches > 1:
            return "patch"
        return "unknown"

    if n_features == 1 and n_patches > 1:
        return "slide"

    if n_features > 1 and n_features != n_patches:
        return "invalid"

    return "invalid"


def write_features(
    slide_artifact: FileHandleH5,
    bag_id: str,
    extractor_name: str,
    feature_matrix: np.ndarray,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
    """Write a feature matrix shaped ``[N, D]`` as ``float32``."""
    feature_array = np.asarray(feature_matrix)
    if feature_array.ndim != 2:
        raise ValueError(f"feature_matrix must have shape (N,D). Got {feature_array.shape}.")
    write_array_dataset(
        slide_artifact.h5,
        layout.features_dataset(bag_id, extractor_name),
        feature_array,
        dtype=np.float32,
    )

def delete_all_features_for_bag(
    slide_artifact: FileHandleH5,
    bag_id: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
    """Delete the full features group for one bag when it exists."""
    features_group_path = layout.features_group(bag_id)
    if features_group_path in slide_artifact.h5:
        del slide_artifact.h5[features_group_path]


def _validate_features_shape(
    shape: tuple[int, ...] | None,
    *,
    expected_rows: int | None = None,
) -> None:
    if not shape or len(shape) != 2:
        raise ValueError(f"features must have shape (N,D). Got {shape}.")
    if expected_rows is not None and int(shape[0]) != int(expected_rows):
        raise ValueError(
            f"Features rows must match expected rows: expected {expected_rows}, got {shape[0]}."
        )
