"""Shared helpers for realistic Hugging Face-backed smoke tests.

These helpers keep the smoke suite compact while preserving a clear separation
between:

- data acquisition and caching
- artifact inspection / reuse
- runtime metric capture

The helpers intentionally avoid importing optional heavy dependencies at module
import time so that smoke tests can skip cleanly when the runtime environment
does not include the PathBench extras.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
from typing import Any, Iterator

import numpy as np

from pathbench.core.io.h5.base import FileHandleH5
from pathbench.core.io.h5 import features as features_io
from pathbench.core.io.h5 import tiles as tiles_io


HF_REPO_ID = "RendeiroLab/LazySlide-data"
SMALL_WSI_FILES: tuple[str, ...] = (
    "sample.svs",
    "GTEX-1117F-0526.svs",
    "lung_carcinoma.ndpi",
)
SURVIVAL_FILES: tuple[str, ...] = (
    "TCGA_READ_subset_TITAN.h5ad",
    "TCGA_READ_survival.csv",
)


@dataclass(frozen=True)
class DownloadedSmokeAssets:
    """Paths to cached smoke-test assets downloaded from Hugging Face.

    Attributes:
        cache_dir: Root directory used for persistent smoke-test downloads.
        slides: Mapping from slide filename to downloaded slide path.
        survival_h5ad: Precomputed TCGA READ feature matrix shaped
            ``[num_slides, feature_dim]``.
        survival_csv: TCGA READ metadata CSV containing survival columns such as
            ``OS_MONTHS`` and ``OS_STATUS``.
    """

    cache_dir: Path
    slides: dict[str, Path]
    survival_h5ad: Path
    survival_csv: Path


@dataclass(frozen=True)
class ExtractedWsiWorkspace:
    """Reusable WSI feature-extraction outputs for the smoke suite.

    Attributes:
        root_dir: Session-scoped writable workspace.
        slides_dir: Directory containing local slide copies or links.
        artifacts_dir: Directory containing per-slide H5 outputs.
        annotations_csv: Annotation CSV consumed by ``WSIDataset``.
        artifact_paths: Mapping from slide stem to PathBench slide artifact.
        bag_id: Bag namespace inside each H5 artifact, for example
            ``"224px_1mpp"``.
        extractor_name: Tile-level feature extractor name used for extraction.
        metrics_path: JSON sidecar containing timing and memory metrics for the
            extraction step.
    """

    root_dir: Path
    slides_dir: Path
    artifacts_dir: Path
    annotations_csv: Path
    artifact_paths: dict[str, Path]
    bag_id: str
    extractor_name: str
    metrics_path: Path


@dataclass(frozen=True)
class PreparedBagWorkspace:
    """Prepared MIL bags backed by ``.pt`` tensors and a metadata CSV.

    Attributes:
        root_dir: Session-scoped writable workspace for prepared bag files.
        feature_dir: Directory containing ``{slide_id}.pt`` bags.
        metadata_csv: Metadata table containing at least ``slide_id`` and one or
            more target columns.
        slide_ids: Ordered slide identifiers with matching bag files.
        input_dim: Feature dimension ``D`` for each bag tensor shaped
            ``[num_instances, D]``.
        bag_lengths: Mapping from slide id to number of instances per bag.
        metrics_path: JSON sidecar containing timing and memory metrics for the
            preparation step.
    """

    root_dir: Path
    feature_dir: Path
    metadata_csv: Path
    slide_ids: list[str]
    input_dim: int
    bag_lengths: dict[str, int]
    metrics_path: Path


def default_smoke_cache_dir() -> Path:
    """Return the default persistent cache directory for smoke-test downloads."""
    return Path(
        os.environ.get("PATHBENCH_SMOKE_CACHE", "~/.cache/pathbench_smoke")
    ).expanduser()


def download_smoke_assets(cache_dir: Path) -> DownloadedSmokeAssets:
    """Download and cache the small smoke-test dataset subset.

    Args:
        cache_dir: Persistent cache directory receiving downloaded files.

    Returns:
        DownloadedSmokeAssets: Resolved local paths to the downloaded assets.
    """

    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    slides = {
        filename: Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=str(cache_dir),
            )
        )
        for filename in SMALL_WSI_FILES
    }
    survival_h5ad = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=SURVIVAL_FILES[0],
            repo_type="dataset",
            local_dir=str(cache_dir),
        )
    )
    survival_csv = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=SURVIVAL_FILES[1],
            repo_type="dataset",
            local_dir=str(cache_dir),
        )
    )
    return DownloadedSmokeAssets(
        cache_dir=cache_dir,
        slides=slides,
        survival_h5ad=survival_h5ad,
        survival_csv=survival_csv,
    )


def link_or_copy(src: Path, dst: Path) -> Path:
    """Materialize ``src`` at ``dst`` via hardlink, symlink, or file copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    try:
        os.link(src, dst)
        return dst
    except OSError:
        pass
    try:
        dst.symlink_to(src)
        return dst
    except OSError:
        shutil.copy2(src, dst)
        return dst


def read_h5_feature_matrix(
    artifact_path: Path,
    *,
    bag_id: str,
    extractor_name: str,
) -> np.ndarray:
    """Load one tile-level feature matrix from a slide H5 artifact.

    Args:
        artifact_path: PathBench slide artifact path.
        bag_id: Bag namespace such as ``"224px_1mpp"``.
        extractor_name: Feature extractor registry key.

    Returns:
        np.ndarray: Feature matrix shaped ``[num_tiles, feature_dim]`` with
        dtype ``float32``.
    """

    with FileHandleH5(artifact_path, mode="r") as slide_artifact:
        features = features_io.read_features(slide_artifact, bag_id, extractor_name)
    return np.asarray(features, dtype=np.float32)


def read_h5_coords(artifact_path: Path, *, bag_id: str) -> np.ndarray:
    """Load one bag coordinate matrix from a slide H5 artifact."""
    with FileHandleH5(artifact_path, mode="r") as slide_artifact:
        coords = tiles_io.read_coords(slide_artifact, bag_id)
    return np.asarray(coords, dtype=np.int32)


def aggregate_slide_feature_matrix(
    artifact_paths: dict[str, Path],
    *,
    bag_id: str,
    extractor_name: str,
) -> tuple[list[str], np.ndarray]:
    """Aggregate tile features into compact slide-level features.

    The smoke suite uses deterministic mean and max pooling to derive a
    slide-level representation from PathBench tile artifacts without rerunning
    feature extraction.

    Args:
        artifact_paths: Mapping from slide id to slide artifact path.
        bag_id: Bag namespace such as ``"224px_1mpp"``.
        extractor_name: Feature extractor registry key stored in the H5 file.

    Returns:
        Tuple containing the ordered slide ids and a slide feature matrix shaped
        ``[num_slides, 2 * feature_dim]``.
    """

    slide_ids = sorted(artifact_paths)
    pooled_rows: list[np.ndarray] = []
    for slide_id in slide_ids:
        tile_features = read_h5_feature_matrix(
            artifact_paths[slide_id],
            bag_id=bag_id,
            extractor_name=extractor_name,
        )
        pooled_rows.append(
            np.concatenate(
                [
                    tile_features.mean(axis=0, dtype=np.float32),
                    tile_features.max(axis=0),
                ],
                axis=0,
            ).astype(np.float32, copy=False)
        )
    return slide_ids, np.stack(pooled_rows, axis=0)


def save_slide_feature_matrix(
    output_path: Path,
    *,
    slide_ids: list[str],
    slide_features: np.ndarray,
) -> Path:
    """Persist slide-level smoke features plus metadata JSON sidecar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, slide_features.astype(np.float32, copy=False))
    metadata = {
        "slide_ids": slide_ids,
        "shape": list(slide_features.shape),
        "dtype": str(slide_features.dtype),
        "aggregation": ["mean", "max"],
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def smoke_metrics_filename(step_name: str) -> str:
    """Normalize a smoke step name into a JSON metrics filename."""
    safe_name = step_name.replace(" ", "_").replace("/", "_")
    return f"{safe_name}.metrics.json"


def _ru_maxrss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


@contextmanager
def capture_smoke_metrics(
    output_dir: Path,
    *,
    step_name: str,
    metadata: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Capture simple timing and memory metrics for one smoke step.

    Args:
        output_dir: Directory receiving the JSON metrics artifact.
        step_name: Human-readable step name.
        metadata: Optional initial metadata stored alongside the measured stats.

    Yields:
        Mutable dictionary that callers can extend with additional metadata
        before the JSON sidecar is persisted.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = dict(metadata or {})
    start = time.perf_counter()
    start_rss_mb = _ru_maxrss_mb()
    try:
        yield payload
    finally:
        payload.update(
            {
                "step_name": step_name,
                "elapsed_seconds": round(time.perf_counter() - start, 6),
                "ru_maxrss_mb": round(_ru_maxrss_mb(), 3),
                "ru_maxrss_delta_mb": round(_ru_maxrss_mb() - start_rss_mb, 3),
            }
        )
        metrics_path = output_dir / smoke_metrics_filename(step_name)
        metrics_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
