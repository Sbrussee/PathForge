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


def prediction_heatmap_exists(
    slide_artifact: FileHandleH5,
    bag_id: str,
    heatmap_name: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> bool:
    """Return whether a prediction heatmap score dataset exists."""

    return exists(slide_artifact.h5, layout.prediction_heatmap_scores_dataset(bag_id, heatmap_name))


def write_prediction_heatmap(
    slide_artifact: FileHandleH5,
    bag_id: str,
    heatmap_name: str,
    *,
    coords: np.ndarray,
    scores: np.ndarray,
    metadata: dict[str, Any] | None = None,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
    """Persist a prediction heatmap under the dedicated H5 prediction namespace.

    Args:
        slide_artifact: Open H5 handle for one slide artifact.
        bag_id: Tiling/feature bag id, for example ``"256px_0.5mpp"``.
        heatmap_name: Non-empty heatmap identifier without ``"/"``.
        coords: Coordinate array shaped ``(N, 2)``. Values are x/y tile
            coordinates in level-0 pixels or another coordinate space documented
            by ``metadata``.
        scores: Finite normalized heatmap scores shaped ``(N,)`` with values in
            ``[0, 1]``.
        metadata: JSON-serializable metadata such as backend name, model path,
            source score file, and coordinate space.

    Raises:
        ValueError: If shapes differ, values are non-finite, or scores are
            outside ``[0, 1]``.
    """

    coords_array = np.asarray(coords)
    scores_array = np.asarray(scores, dtype=np.float32)
    if coords_array.ndim != 2 or coords_array.shape[1] != 2:
        raise ValueError(f"heatmap coords must have shape (N,2). Got {coords_array.shape}.")
    if scores_array.ndim != 1:
        raise ValueError(f"heatmap scores must have shape (N,). Got {scores_array.shape}.")
    if coords_array.shape[0] != scores_array.shape[0]:
        raise ValueError(
            "heatmap coords and scores must have the same row count. "
            f"Got {coords_array.shape[0]} and {scores_array.shape[0]}."
        )
    if not np.isfinite(scores_array).all():
        raise ValueError("heatmap scores contain NaN or Inf.")
    if np.any(scores_array < 0.0) or np.any(scores_array > 1.0):
        raise ValueError("heatmap scores must be normalized to [0, 1].")

    write_array_dataset(
        slide_artifact.h5,
        layout.prediction_heatmap_coords_dataset(bag_id, heatmap_name),
        coords_array,
        dtype=np.float32,
    )
    write_array_dataset(
        slide_artifact.h5,
        layout.prediction_heatmap_scores_dataset(bag_id, heatmap_name),
        scores_array,
        dtype=np.float32,
    )
    write_json_dataset(
        slide_artifact.h5,
        layout.prediction_heatmap_metadata_dataset(bag_id, heatmap_name),
        metadata or {},
    )


def read_prediction_heatmap(
    slide_artifact: FileHandleH5,
    bag_id: str,
    heatmap_name: str,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> dict[str, Any]:
    """Read a persisted prediction heatmap from an open slide H5 artifact."""

    coords = read_array_dataset(slide_artifact.h5, layout.prediction_heatmap_coords_dataset(bag_id, heatmap_name))
    scores = read_array_dataset(slide_artifact.h5, layout.prediction_heatmap_scores_dataset(bag_id, heatmap_name))
    metadata = read_json_dataset(slide_artifact.h5, layout.prediction_heatmap_metadata_dataset(bag_id, heatmap_name))
    return {"coords": coords, "scores": scores, "metadata": metadata}
