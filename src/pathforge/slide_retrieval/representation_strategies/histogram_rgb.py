"""Row-aligned RGB histograms and SISH quality descriptors."""

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
from pathforge.slide_retrieval.annotations import resolve_sample_annotations
from pathforge.slide_retrieval.representation_strategies.mean_rgb import (
    _build_slide_processor,
    _resolve_sample_slide_paths,
    _slide_retrieval_artifact_path,
)

HISTOGRAM_RGB_DESCRIPTOR_NAME = "histogram_rgb"
HISTOGRAM_RGB_DIM = 768
HISTOGRAM_RGB_SIZE = 256
_CONTRACT = {"bins_per_channel": 256, "resize_px": 256, "channels": "rgb", "version": 1}
SISH_QUALITY_DESCRIPTOR_NAME = "sish_quality"
SISH_QUALITY_DIM = 129
_QUALITY_CONTRACT = {
    "resize_px": 256,
    "lbp_points": 8,
    "lbp_radius": 1,
    "lbp_method": "ror",
    "lbp_bins": 128,
    "lbp_range_max": 128,
    "white_threshold": 235,
    "preprocessing": "opencv",
    "version": 1,
}


def resolve_sample_patch_histogram_rgb(
    *, sample: Any, bag_id: str, config: Any
) -> np.ndarray:
    """Load or create SISH RGB histograms, aligned to patch-feature rows.

    The result has shape ``(N, 768)``: three concatenated 256-bin channel
    histograms for each stored tile. Missing descriptors are materialized from
    source WSIs and persisted in each slide retrieval artifact.
    """
    return resolve_sample_patch_rgb_descriptors(
        sample=sample, bag_id=bag_id, config=config
    )[HISTOGRAM_RGB_DESCRIPTOR_NAME]


def resolve_sample_patch_rgb_descriptors(
    *, sample: Any, bag_id: str, config: Any, include_quality: bool = False
) -> dict[str, np.ndarray]:
    """Load or create colour and optionally quality descriptors in tile-row order.

    Args:
        sample: Slide IDs and their aligned artifact paths.
        bag_id: Canonical tiling ID identifying the stored coordinates.
        config: Slide backend and dataset configuration; source slides are needed
            only when required cache entries are missing or invalid.
        include_quality: Require the SISH quality descriptor in addition to RGB.

    Returns:
        ``histogram_rgb`` with shape ``(N, 768)`` and, when requested,
        ``sish_quality`` with shape ``(N, 129)``. The quality matrix contains
        128 density-normalized LBP bins followed by the white-pixel fraction.
        All matrices have dtype ``float32`` and match the stored tile rows.

    Example:
        >>> descriptors = resolve_sample_patch_rgb_descriptors(
        ...     sample=sample, bag_id="256px_0.5mpp", config=cfg,
        ...     include_quality=True,
        ... )
        >>> descriptors["sish_quality"].shape[1]
        129
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
    annotation_rows = None
    processor = None
    contracts = {HISTOGRAM_RGB_DESCRIPTOR_NAME: (HISTOGRAM_RGB_DIM, _CONTRACT)}
    if include_quality:
        contracts[SISH_QUALITY_DESCRIPTOR_NAME] = (SISH_QUALITY_DIM, _QUALITY_CONTRACT)
    parts: dict[str, list[np.ndarray]] = {name: [] for name in contracts}
    try:
        for slide_id, artifact_path in zip(slide_ids, artifact_paths, strict=True):
            with FileHandleH5(artifact_path, mode="r") as artifact:
                coords = tiles_io.read_coords(artifact, bag_id=bag_id)
            target = _slide_retrieval_artifact_path(
                slide_artifact_path=artifact_path, slide_id=slide_id
            )
            cached: dict[str, np.ndarray] = {}
            if target.is_file():
                with FileHandleH5(target, mode="r") as retrieval:
                    for name, (dim, contract) in contracts.items():
                        if descriptors_io.descriptor_exists(
                            retrieval,
                            tile_id=bag_id,
                            descriptor_name=name,
                            expected_rows=len(coords),
                            expected_dim=dim,
                            expected_metadata=contract,
                        ):
                            cached[name] = descriptors_io.read_descriptor(
                                retrieval, bag_id, name
                            )
            if len(cached) == len(contracts):
                for name, matrix in cached.items():
                    parts[name].append(matrix)
                continue
            # Empty bags need no image reads, even when their caches are absent.
            if len(coords) == 0:
                matrices = {
                    name: np.empty((0, dim), dtype=np.float32)
                    for name, (dim, _) in contracts.items()
                }
            else:
                if paths is None:
                    paths = _resolve_sample_slide_paths(sample=sample, config=config)
                slide_path = paths.get(slide_id)
                if slide_path is None or not slide_path.is_file():
                    missing = ", ".join(name for name in contracts if name not in cached)
                    raise FileNotFoundError(
                        f"Missing stored {missing} descriptors and no source slide is available "
                        f"for slide '{slide_id}' at bag_id='{bag_id}'."
                    )
                if processor is None:
                    processor = _build_slide_processor(config=config)
                if annotation_rows is None:
                    annotation_rows = resolve_sample_annotations(sample=sample, config=config)
                matrices = _create_slide_descriptors(
                    slide_id,
                    slide_path,
                    artifact_path,
                    coords,
                    processor,
                    include_quality=include_quality,
                    annotation_row=annotation_rows[slide_id],
                )
            with atomic_slide_artifact_write(target) as retrieval:
                for name, (_, contract) in contracts.items():
                    # Preserve valid RGB caches when upgrading a legacy artifact.
                    matrix = cached.get(name, matrices[name])
                    if name not in cached:
                        descriptors_io.write_descriptor(
                            retrieval, bag_id, name, matrix, metadata=contract
                        )
                    parts[name].append(matrix)
    finally:
        close = getattr(processor, "close", None)
        if callable(close):
            close()
    return {name: np.concatenate(rows, axis=0) for name, rows in parts.items()}


def _create_slide_descriptors(
    slide_id: str,
    slide_path: Path,
    artifact_path: Path,
    coords: np.ndarray,
    processor: Any,
    *,
    include_quality: bool,
    annotation_row: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Read ``(N, 5)`` tile coordinates once to build RGB and quality matrices.

    Returns ``histogram_rgb: (N, 768)`` and optionally ``sish_quality: (N, 129)``
    as float32 arrays. Grayscale preprocessing matches SISH's existing filter.

    Example:
        >>> matrices = _create_slide_descriptors(
        ...     slide_id, slide_path, artifact_path, coords, processor,
        ...     include_quality=True,
        ... )
    """
    if include_quality:
        from skimage.feature import local_binary_pattern

    wsi = WSI.from_annotation(
        annotation_row if annotation_row is not None else {"slide": slide_id},
        slide_path=slide_path,
        artifact_path=artifact_path,
    )
    processor.load_wsi(wsi)
    try:
        rows = np.empty((len(coords), HISTOGRAM_RGB_DIM), dtype=np.float32)
        quality = np.empty((len(coords), SISH_QUALITY_DIM), dtype=np.float32)
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
            if include_quality:
                # Convert before resizing to preserve the previous filtering input.
                grey = cv.resize(cv.cvtColor(patch, cv.COLOR_RGB2GRAY), (256, 256))
                lbp = local_binary_pattern(grey, 8, 1, "ror")
                quality[index, :128] = np.histogram(
                    lbp, density=True, bins=128, range=(0, 128)
                )[0]
                quality[index, 128] = np.mean(grey > 235)
            patch = cv.resize(patch, (HISTOGRAM_RGB_SIZE, HISTOGRAM_RGB_SIZE))
            rows[index] = np.concatenate(
                [
                    cv.calcHist([patch], [channel], None, [256], [0, 256]).reshape(-1)
                    for channel in range(3)
                ]
            ).astype(np.float32)
    finally:
        processor.close_wsi(wsi)
    matrices = {HISTOGRAM_RGB_DESCRIPTOR_NAME: rows}
    if include_quality:
        matrices[SISH_QUALITY_DESCRIPTOR_NAME] = quality
    return matrices
