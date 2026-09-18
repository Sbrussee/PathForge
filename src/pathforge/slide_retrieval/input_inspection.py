"""Cheap structural classification of original retrieval feature artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pathforge.core.io.slide_artifacts import tiles as tiles_io
from pathforge.core.io.slide_artifacts.base import (
    FileHandleH5,
    get_dataset,
    is_complete,
)
from pathforge.core.io.slide_artifacts.layout import DEFAULT_LAYOUT

FeatureLevel = Literal["patch", "slide"]


@dataclass(frozen=True, slots=True)
class RetrievalInputInspection:
    """Original-feature family and common vector width for one retrieval run."""

    feature_level: FeatureLevel
    feature_dimension: int


def inspect_retrieval_inputs(
    artifact_paths: Iterable[Path],
    *,
    tiling_id: str,
    extractor_name: str,
) -> RetrievalInputInspection:
    """Classify original H5 features without materializing feature matrices.

    One feature row is a slide vector. Multiple rows are patch features only
    when their count equals the coordinate-row count. All inspected artifacts
    must share that family and their positive feature width.

    Example:
        >>> inspection = inspect_retrieval_inputs([], tiling_id="tiles", extractor_name="uni")
        Traceback (most recent call last):
        ...
        ValueError: Cannot inspect retrieval inputs without physical-slide artifacts.
    """
    observed_level: FeatureLevel | None = None
    observed_dimension: int | None = None
    inspected_paths: set[Path] = set()
    for raw_path in artifact_paths:
        artifact_path = Path(raw_path)
        if artifact_path in inspected_paths:
            continue
        inspected_paths.add(artifact_path)
        feature_shape, coordinate_rows = _inspect_artifact_shape(
            artifact_path,
            tiling_id=tiling_id,
            extractor_name=extractor_name,
        )
        rows, dimension = feature_shape
        if rows <= 0 or dimension <= 0 or coordinate_rows <= 0:
            raise ValueError(
                f"Invalid empty retrieval input in {artifact_path.name}: "
                f"feature_shape={feature_shape}, coordinate_rows={coordinate_rows}."
            )
        level: FeatureLevel
        if rows == 1:
            level = "slide"
        elif rows == coordinate_rows:
            level = "patch"
        else:
            raise ValueError(
                f"Misaligned retrieval input in {artifact_path.name}: "
                f"feature_rows={rows}, coordinate_rows={coordinate_rows}."
            )
        if observed_level is not None and level != observed_level:
            raise ValueError("A retrieval run cannot contain mixed patch and slide feature inputs.")
        if observed_dimension is not None and dimension != observed_dimension:
            raise ValueError(
                "A retrieval run requires one feature dimension; "
                f"got {observed_dimension} and {dimension}."
            )
        observed_level = level
        observed_dimension = dimension

    if observed_level is None or observed_dimension is None:
        raise ValueError("Cannot inspect retrieval inputs without physical-slide artifacts.")
    return RetrievalInputInspection(observed_level, observed_dimension)


def _inspect_artifact_shape(
    artifact_path: Path,
    *,
    tiling_id: str,
    extractor_name: str,
) -> tuple[tuple[int, int], int]:
    """Read feature and coordinate shapes only for one physical-slide H5."""
    with FileHandleH5(artifact_path, mode="r") as artifact:
        feature_path = DEFAULT_LAYOUT.features_dataset(tiling_id, extractor_name)
        dataset = get_dataset(artifact.h5, feature_path)
        if dataset is None or not is_complete(dataset):
            raise ValueError(f"Missing or incomplete features dataset: {feature_path}.")
        shape = tuple(int(value) for value in dataset.shape)
        if len(shape) != 2:
            raise ValueError(f"Features must have shape (N, D); got {shape}.")
        return (shape[0], shape[1]), tiles_io.coords_num_rows(artifact, bag_id=tiling_id)
