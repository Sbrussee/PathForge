"""Row-aligned SISH RGB-histogram descriptors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from pathforge.core.datasets.wsi_dataset import WSI
from pathforge.core.io.slide_artifacts import tiles as tiles_io
from pathforge.core.io.slide_artifacts.atomic import atomic_slide_artifact_write
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.core.io.slide_retrieval import descriptors as descriptors_io
from pathforge.slide_retrieval.representation_strategies.mean_rgb import (
    _build_slide_processor,
    _resolve_sample_slide_paths,
    _slide_retrieval_artifact_path,
)

HISTOGRAM_RGB_DESCRIPTOR_NAME = "histogram_rgb"
HISTOGRAM_RGB_DIM = 768
HISTOGRAM_RGB_SIZE = 256
_CONTRACT = {"bins_per_channel": 256, "resize_px": 256, "channels": "rgb", "version": 1}


def resolve_sample_patch_histogram_rgb(
    *, sample: Any, bag_id: str, config: Any
) -> np.ndarray:
    """Load or create SISH RGB histograms, aligned to patch-feature rows.

    The result has shape ``(N, 768)``: three concatenated 256-bin channel
    histograms for each stored tile. Missing descriptors are materialized from
    source WSIs and persisted in each slide retrieval artifact.
    """
    if sample is None or config is None:
        raise ValueError(
            "sample and config are required to resolve histogram_rgb descriptors."
        )
    artifact_paths = [
        Path(path) for path in list(getattr(sample, "artifact_paths", []) or [])
    ]
    slide_ids = [str(item) for item in list(getattr(sample, "slide_ids", []) or [])]
    if not artifact_paths or len(artifact_paths) != len(slide_ids):
        raise ValueError(
            "sample.slide_ids and sample.artifact_paths must be non-empty and aligned."
        )
    paths: dict[str, Path | None] | None = None
    processor = None
    parts: list[np.ndarray] = []
    try:
        for slide_id, artifact_path in zip(slide_ids, artifact_paths, strict=True):
            with FileHandleH5(artifact_path, mode="r") as artifact:
                coords = tiles_io.read_coords(artifact, bag_id=bag_id)
            target = _slide_retrieval_artifact_path(
                slide_artifact_path=artifact_path, slide_id=slide_id
            )
            if target.is_file():
                with FileHandleH5(target, mode="r") as retrieval:
                    if descriptors_io.descriptor_exists(
                        retrieval,
                        tile_id=bag_id,
                        descriptor_name=HISTOGRAM_RGB_DESCRIPTOR_NAME,
                        expected_rows=len(coords),
                        expected_dim=HISTOGRAM_RGB_DIM,
                        expected_metadata=_CONTRACT,
                    ):
                        parts.append(
                            descriptors_io.read_descriptor(
                                retrieval, bag_id, HISTOGRAM_RGB_DESCRIPTOR_NAME
                            )
                        )
                        continue
            if paths is None:
                paths = _resolve_sample_slide_paths(sample=sample, config=config)
            slide_path = paths.get(slide_id)
            if slide_path is None or not slide_path.is_file():
                raise FileNotFoundError(
                    f"Missing stored histogram_rgb descriptors and no source slide is available for slide '{slide_id}' at bag_id='{bag_id}'."
                )
            if processor is None:
                processor = _build_slide_processor(config=config)
            matrix = _create_slide_histograms(
                slide_id, slide_path, artifact_path, coords, processor
            )
            with atomic_slide_artifact_write(target) as retrieval:
                descriptors_io.write_descriptor(
                    retrieval,
                    bag_id,
                    HISTOGRAM_RGB_DESCRIPTOR_NAME,
                    matrix,
                    metadata=_CONTRACT,
                )
            parts.append(matrix)
    finally:
        close = getattr(processor, "close", None)
        if callable(close):
            close()
    return (
        np.concatenate(parts, axis=0)
        if parts
        else np.empty((0, HISTOGRAM_RGB_DIM), dtype=np.float32)
    )


def _create_slide_histograms(
    slide_id: str,
    slide_path: Path,
    artifact_path: Path,
    coords: np.ndarray,
    processor: Any,
) -> np.ndarray:
    """Read one WSI and return its row-aligned ``(N, 768)`` histogram matrix."""
    wsi = WSI(
        slide=slide_id,
        patient="",
        category="",
        path=slide_path,
        artifact_path=artifact_path,
    )
    processor.load_wsi(wsi)
    try:
        rows = np.empty((len(coords), HISTOGRAM_RGB_DIM), dtype=np.float32)
        for index, row in enumerate(np.asarray(coords, dtype=np.int32)):
            x, y, width, height, level = (int(value) for value in row)
            patch = np.asarray(
                processor.read_patch_region(
                    wsi, x=x, y=y, width=width, height=height, level=level
                ),
                dtype=np.uint8,
            )
            if patch.ndim != 3 or patch.shape[2] != 3:
                raise ValueError(
                    f"Patch RGB reads must have shape (H,W,3). Got {patch.shape} for slide '{slide_id}'."
                )
            patch = cv.resize(patch, (HISTOGRAM_RGB_SIZE, HISTOGRAM_RGB_SIZE))
            rows[index] = np.concatenate(
                [
                    cv.calcHist([patch], [channel], None, [256], [0, 256]).reshape(-1)
                    for channel in range(3)
                ]
            ).astype(np.float32)
    finally:
        processor.close_wsi(wsi)
    return rows
