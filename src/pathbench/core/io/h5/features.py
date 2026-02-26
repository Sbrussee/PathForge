# src/pathbench/core/io/h5/features.py
from __future__ import annotations

import numpy as np

from pathbench.core.io.h5.base import FileHandleH5, exists, read_array_dataset, write_array_dataset
from pathbench.core.io.h5.layout import DEFAULT_LAYOUT, H5Layout


def features_exist(
    slide_artifact: FileHandleH5,
    bag_id: str,
    extractor_name: str,
    *,
    expected_rows: int | None = None,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> bool:
    path = layout.features_dataset(bag_id, extractor_name)
    if not exists(slide_artifact.h5, path):
        return False
    if expected_rows is None:
        return True
    dset = slide_artifact.h5[path]
    shape = getattr(dset, "shape", None)
    return bool(shape and len(shape) == 2 and int(shape[0]) == int(expected_rows))


def read_features(
    slide_artifact: FileHandleH5,
    bag_id: str,
    extractor_name: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> np.ndarray:
    feature_matrix = read_array_dataset(slide_artifact.h5, layout.features_dataset(bag_id, extractor_name))
    if feature_matrix.ndim != 2:
        raise ValueError(f"features must have shape (N,D). Got {feature_matrix.shape}.")
    return feature_matrix.astype(np.float32, copy=False)


def write_features(
    slide_artifact: FileHandleH5,
    bag_id: str,
    extractor_name: str,
    feature_matrix: np.ndarray,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
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
    features_group_path = layout.features_group(bag_id)
    if features_group_path in slide_artifact.h5:
        del slide_artifact.h5[features_group_path]
