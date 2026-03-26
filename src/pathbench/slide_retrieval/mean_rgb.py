from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from pathbench.core.datasets.wsi_dataset import WSI
from pathbench.core.io.h5 import descriptors as descriptors_io
from pathbench.core.io.h5 import tiles as tiles_io
from pathbench.core.io.h5.base import FileHandleH5
from pathbench.core.slide_processing.base import SlideProcessorBase
from pathbench.utils.constants import SLIDE_FILE_FORMATS
from pathbench.utils.registries import SLIDE_PROCESSORS


MEAN_RGB_DESCRIPTOR_NAME = "mean_rgb"


def resolve_sample_patch_mean_rgb(
    *,
    sample: Any,
    bag_id: str,
    config: Any,
) -> np.ndarray:
    """
    Load or materialize row-aligned patch mean RGB descriptors for one sample.

    Inputs:
    - `sample`: sample-like object exposing `slide_ids`, `artifact_paths`, and
      `metadata`.
    - `bag_id`: canonical tiling identifier used in the slide H5 layout.
    - `config`: config-like object exposing `slide_processing.backend` and
      `datasets`.

    Returns:
    - `np.ndarray[float32]` with shape `(N, 3)` aligned with the retrieval bag.

    Example:
    ```python
    mean_rgb = resolve_sample_patch_mean_rgb(
        sample=sample,
        bag_id="256px_0.5mpp",
        config=cfg,
    )
    ```
    """
    if sample is None:
        raise ValueError("sample is required to resolve patch mean RGB descriptors.")
    if config is None:
        raise ValueError("config is required to resolve patch mean RGB descriptors.")

    artifact_paths = [Path(path) for path in list(getattr(sample, "artifact_paths", []) or [])]
    slide_ids = [str(slide_id) for slide_id in list(getattr(sample, "slide_ids", []) or [])]

    if not artifact_paths:
        raise ValueError("sample.artifact_paths is required to resolve patch mean RGB descriptors.")
    if len(slide_ids) != len(artifact_paths):
        raise ValueError(
            "sample.slide_ids and sample.artifact_paths must have the same length. "
            f"Got {len(slide_ids)} and {len(artifact_paths)}."
        )

    mean_rgb_parts: list[np.ndarray] = []
    slide_paths_by_id: dict[str, Path | None] | None = None
    slide_processor: SlideProcessorBase | None = None
    for slide_id, artifact_path in zip(slide_ids, artifact_paths):
        with FileHandleH5(artifact_path, mode="r") as slide_artifact:
            coords = tiles_io.read_coords(slide_artifact, bag_id=bag_id)
            expected_rows = int(coords.shape[0])
            if descriptors_io.descriptor_exists(
                slide_artifact,
                bag_id=bag_id,
                descriptor_name=MEAN_RGB_DESCRIPTOR_NAME,
                expected_rows=expected_rows,
                expected_dim=3,
            ):
                mean_rgb_parts.append(
                    descriptors_io.read_descriptor(
                        slide_artifact,
                        bag_id=bag_id,
                        descriptor_name=MEAN_RGB_DESCRIPTOR_NAME,
                    )
                )
                continue

        if slide_paths_by_id is None:
            slide_paths_by_id = _resolve_sample_slide_paths(sample=sample, config=config)
        if slide_processor is None:
            slide_processor = _build_slide_processor(config=config)

        mean_rgb_parts.append(
            _resolve_slide_patch_mean_rgb(
                artifact_path=artifact_path,
                slide_path=slide_paths_by_id.get(slide_id),
                bag_id=bag_id,
                slide_processor=slide_processor,
                slide_id=slide_id,
            )
        )

    if not mean_rgb_parts:
        return np.empty((0, 3), dtype=np.float32)

    return np.concatenate(mean_rgb_parts, axis=0).astype(np.float32, copy=False)


def _resolve_slide_patch_mean_rgb(
    *,
    artifact_path: Path,
    slide_path: Path | None,
    bag_id: str,
    slide_processor: SlideProcessorBase,
    slide_id: str,
) -> np.ndarray:
    """
    Resolve one slide-level mean RGB descriptor matrix from H5 or source slide.

    Inputs:
    - `artifact_path`: H5 artifact path for the slide.
    - `slide_path`: source slide path used to compute missing descriptors.
    - `bag_id`: canonical tiling identifier.
    - `slide_processor`: backend implementation used to read patch pixels.
    - `slide_id`: human-readable slide identifier for error messages.

    Returns:
    - `np.ndarray[float32]` with shape `(N, 3)`.
    """
    with FileHandleH5(artifact_path, mode="a") as slide_artifact:
        if descriptors_io.descriptor_exists(
            slide_artifact,
            bag_id=bag_id,
            descriptor_name=MEAN_RGB_DESCRIPTOR_NAME,
            expected_rows=tiles_io.coords_num_rows(slide_artifact, bag_id=bag_id),
            expected_dim=3,
        ):
            return descriptors_io.read_descriptor(
                slide_artifact,
                bag_id=bag_id,
                descriptor_name=MEAN_RGB_DESCRIPTOR_NAME,
            )

        if slide_path is None or not slide_path.is_file():
            raise FileNotFoundError(
                "Missing stored patch mean RGB descriptors and no source slide is "
                f"available to compute them for slide '{slide_id}' at bag_id='{bag_id}'."
            )

        mean_rgb = _create_slide_patch_mean_rgb(
            slide_artifact=slide_artifact,
            slide_path=slide_path,
            bag_id=bag_id,
            slide_processor=slide_processor,
            slide_id=slide_id,
        )
        descriptors_io.write_descriptor(
            slide_artifact,
            bag_id=bag_id,
            descriptor_name=MEAN_RGB_DESCRIPTOR_NAME,
            descriptor_matrix=mean_rgb,
        )
        return mean_rgb


def _create_slide_patch_mean_rgb(
    *,
    slide_artifact: FileHandleH5,
    slide_path: Path,
    bag_id: str,
    slide_processor: SlideProcessorBase,
    slide_id: str,
) -> np.ndarray:
    """
    Compute one slide-level mean RGB descriptor matrix from the source slide.

    Inputs:
    - `slide_artifact`: open H5 file handle containing `coords` and `tiling_spec`.
    - `slide_path`: source slide path used to read patch pixels.
    - `bag_id`: canonical tiling identifier.
    - `slide_processor`: backend implementation used to read patch regions.
    - `slide_id`: human-readable slide identifier for error messages.

    Returns:
    - `np.ndarray[float32]` with shape `(N, 3)`.
    """
    coords = tiles_io.read_coords(slide_artifact, bag_id=bag_id)
    expected_rows = int(coords.shape[0])
    slide_wsi = WSI(
        slide=slide_id,
        patient="",
        category="",
        path=slide_path,
        artifact_path=slide_artifact.path,
    )
    slide_processor.load_wsi(slide_wsi)
    try:
        mean_rgb = np.empty((expected_rows, 3), dtype=np.float32)
        for row_index, row in enumerate(np.asarray(coords, dtype=np.int32)):
            x0, y0, read_w, read_h, read_level = [int(value) for value in row]
            patch = slide_processor.read_patch_region(
                slide_wsi,
                x=x0,
                y=y0,
                width=read_w,
                height=read_h,
                level=read_level,
            )
            patch_array = np.asarray(patch, dtype=np.uint8)
            if patch_array.ndim != 3 or patch_array.shape[2] != 3:
                raise ValueError(
                    "Patch RGB reads must have shape (H,W,3). "
                    f"Got {patch_array.shape} for slide '{slide_id}'."
                )
            mean_rgb[row_index] = (
                patch_array.reshape(-1, 3).mean(axis=0, dtype=np.float64) / 255.0
            ).astype(np.float32, copy=False)
    finally:
        slide_processor.close_wsi(slide_wsi)

    mean_rgb = np.asarray(mean_rgb, dtype=np.float32)
    if mean_rgb.ndim != 2 or mean_rgb.shape != (expected_rows, 3):
        raise ValueError(
            "Patch mean RGB descriptors must have shape (N,3). "
            f"Got {mean_rgb.shape} for slide '{slide_id}'."
        )

    return mean_rgb


def _resolve_sample_slide_paths(
    *,
    sample: Any,
    config: Any,
) -> dict[str, Path | None]:
    """
    Resolve source slide paths for all member slides of one retrieval sample.

    Inputs:
    - `sample`: sample-like object exposing `slide_ids` and metadata containing
      the dataset name.
    - `config`: config-like object exposing `datasets`.

    Returns:
    - `dict[str, Path | None]` mapping slide id to discovered slide path.
    """
    metadata = dict(getattr(sample, "metadata", {}) or {})
    dataset_name = metadata.get("dataset")
    if not dataset_name:
        raise ValueError(
            "sample.metadata['dataset'] is required to resolve source slide paths."
        )

    dataset_cfg = _find_dataset_config(config=config, dataset_name=str(dataset_name))
    slides_dir = Path(_get_config_value(dataset_cfg, "slides_dir")).expanduser().resolve()
    slide_ids = [str(slide_id) for slide_id in list(getattr(sample, "slide_ids", []) or [])]
    return {
        slide_id: _find_slide_path(slides_dir=slides_dir, slide_id=slide_id)
        for slide_id in slide_ids
    }


def _find_dataset_config(*, config: Any, dataset_name: str) -> Any:
    datasets = list(_get_config_value(config, "datasets", []))
    for dataset_cfg in datasets:
        if str(_get_config_value(dataset_cfg, "name")) == dataset_name:
            return dataset_cfg

    raise ValueError(f"Dataset '{dataset_name}' not found in config.datasets.")


def _find_slide_path(*, slides_dir: Path, slide_id: str) -> Path | None:
    candidates = sorted(
        path
        for path in slides_dir.glob(f"{slide_id}.*")
        if path.suffix.lower() in SLIDE_FILE_FORMATS
    )
    if not candidates:
        return None
    return candidates[0]


def _build_slide_processor(*, config: Any) -> SlideProcessorBase:
    slide_processing_cfg = _get_config_value(config, "slide_processing")
    backend_name = str(_get_config_value(slide_processing_cfg, "backend"))

    if not SLIDE_PROCESSORS.is_available(backend_name):
        import_module(f"pathbench.core.slide_processing.{backend_name}")

    processor_cls = SLIDE_PROCESSORS.get(backend_name)
    if processor_cls is None:
        raise ValueError(f"Slide processing backend '{backend_name}' not found in registry.")

    return processor_cls()


def _get_config_value(
    source: Any,
    key: str,
    default: Any = ...,
) -> Any:
    if isinstance(source, dict):
        if key in source:
            return source[key]
    elif hasattr(source, key):
        return getattr(source, key)

    if default is ...:
        raise ValueError(f"Required config value '{key}' is missing.")
    return default
