"""Shared helpers for realistic Hugging Face-backed smoke tests.

These helpers keep the smoke suite compact while preserving a clear separation
between:

- data acquisition and caching
- artifact inspection / reuse
- runtime metric capture

The helpers intentionally avoid importing optional heavy dependencies at module
import time so that smoke tests can skip cleanly when the runtime environment
does not include the PathForge extras.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
from typing import Any, Iterator

import numpy as np

from pathforge.core.io.h5.base import FileHandleH5
from pathforge.core.io.h5 import features as features_io
from pathforge.core.io.h5 import tiles as tiles_io


HF_REPO_ID = "RendeiroLab/LazySlide-data"
SMALL_WSI_FILES: tuple[str, ...] = (
    "sample.svs",
    "gtex_artery_data/GTEX-111YS-2226.svs",
    "gtex_artery_data/GTEX-11GSP-2926.svs",
    "gtex_artery_data/GTEX-11LCK-1426.svs",
    "gtex_artery_data/GTEX-11ONC-2726.svs",
    "gtex_artery_data/GTEX-12126-0726.svs",
    "gtex_artery_data/GTEX-12KS4-1226.svs",
    "gtex_artery_data/GTEX-1339X-2626.svs",
    "gtex_artery_data/GTEX-13O61-2526.svs",
    "gtex_artery_data/GTEX-13RTK-1826.svs",
    "gtex_artery_data/GTEX-13W46-0626.svs",
    "gtex_artery_data/GTEX-144GM-2226.svs",
    "gtex_artery_data/GTEX-14BMU-2426.svs",
    "gtex_artery_data/GTEX-14ICL-1826.svs",
    "gtex_artery_data/GTEX-14PJN-2226.svs",
    "gtex_artery_data/GTEX-15RJE-0526.svs",
    "gtex_artery_data/GTEX-16MT9-0326.svs",
    "gtex_artery_data/GTEX-16NPV-0526.svs",
    "gtex_artery_data/GTEX-16YQH-0626.svs",
    "gtex_artery_data/GTEX-17MF6-0526.svs",
    "gtex_artery_data/GTEX-183FY-0526.svs",
    "gtex_artery_data/GTEX-1C6VS-0326.svs",
    "gtex_artery_data/GTEX-1CAMQ-0426.svs",
    "gtex_artery_data/GTEX-1H23P-0426.svs",
    "gtex_artery_data/GTEX-1HGF4-2826.svs",
    "gtex_artery_data/GTEX-1HUB1-0226.svs",
    "gtex_artery_data/GTEX-1I1GU-1926.svs",
    "gtex_artery_data/GTEX-1NV5F-0426.svs",
    "gtex_artery_data/GTEX-O5YU-0926.svs",
    "gtex_artery_data/GTEX-PW2O-1926.svs",
    "gtex_artery_data/GTEX-S33H-2426.svs",
    "gtex_artery_data/GTEX-S4Q7-1326.svs",
    "gtex_artery_data/GTEX-SNMC-1526.svs",
    "gtex_artery_data/GTEX-T5JW-2126.svs",
    "gtex_artery_data/GTEX-TKQ1-1226.svs",
    "gtex_artery_data/GTEX-U8XE-0626.svs",
    "gtex_artery_data/GTEX-WHSE-1126.svs",
    "gtex_artery_data/GTEX-WOFL-0426.svs",
    "gtex_artery_data/GTEX-WOFM-1726.svs",
    "gtex_artery_data/GTEX-XPT6-2226.svs",
    "gtex_artery_data/GTEX-XUZC-2026.svs",
    "gtex_artery_data/GTEX-Z93T-0426.svs",
    "gtex_artery_data/GTEX-ZP4G-2126.svs",
    "gtex_artery_data/GTEX-ZQUD-1326.svs",
    "gtex_artery_data/GTEX-ZVZP-2726.svs",
    "gtex_artery_data/GTEX-ZYT6-1526.svs",
)
GTEX_ARTERY_SLIDE_IDS: tuple[str, ...] = tuple(
    Path(f).stem for f in SMALL_WSI_FILES if "gtex_artery_data/" in f
)
CLASSIFICATION_METADATA_FILES: tuple[str, ...] = ("GTEx_artery_dataset.csv.gz",)
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
        gtex_artery_csv: Full GTEx artery metadata table used to derive real
            smoke-test annotations.
        survival_h5ad: Precomputed TCGA READ feature matrix shaped
            ``[num_slides, feature_dim]``.
        survival_csv: TCGA READ metadata CSV containing survival columns such as
            ``OS_MONTHS`` and ``OS_STATUS``.
    """

    cache_dir: Path
    slides: dict[str, Path]
    gtex_artery_csv: Path
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
        artifact_paths: Mapping from slide stem to PathForge slide artifact.
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
    configured_path = os.environ.get("PATHFORGE_SMOKE_CACHE")
    if configured_path:
        return Path(configured_path).expanduser()

    home_cache_dir = Path("~/.cache/pathforge_smoke").expanduser()
    home_dir = Path.home()
    if home_dir.exists() and os.access(home_dir, os.W_OK):
        return home_cache_dir

    tmp_root = Path(os.environ.get("TMPDIR", "/tmp")).expanduser()
    return tmp_root / "pathforge_smoke"


def configured_smoke_report_dir() -> Path | None:
    """Return the optional directory used for aggregated smoke reports."""
    raw_value = os.environ.get("PATHFORGE_SMOKE_REPORT_DIR")
    if not raw_value:
        return None
    return Path(raw_value).expanduser()


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
        Path(filename).name: Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=str(cache_dir),
            )
        )
        for filename in SMALL_WSI_FILES
    }
    gtex_artery_csv = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=CLASSIFICATION_METADATA_FILES[0],
            repo_type="dataset",
            local_dir=str(cache_dir),
        )
    )
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
        gtex_artery_csv=gtex_artery_csv,
        survival_h5ad=survival_h5ad,
        survival_csv=survival_csv,
    )


def load_tabular_smoke_metadata(csv_path: Path) -> Any:
    """Load one smoke metadata table, including gzip-compressed CSV files.

    Args:
        csv_path: Metadata CSV path ending in ``.csv`` or ``.csv.gz``.

    Returns:
        pd.DataFrame: Loaded metadata table.
    """

    import pandas as pd

    return pd.read_csv(csv_path)


def build_gtex_smoke_annotations(
    gtex_metadata_csv: Path,
    *,
    slide_ids: list[str],
    strict: bool = True,
) -> Any:
    """Build PathForge-style annotation rows from the full GTEx artery table.

    Args:
        gtex_metadata_csv: Full GTEx artery metadata CSV path.
        slide_ids: Slide ids without filename extensions.
        strict: When true, require a direct sample-level metadata match for
            every requested slide. When false, fall back to dataset-level GTEx
            artery metadata if the downloadable WSI subset and table do not
            share exact sample ids.

    Returns:
        pd.DataFrame: Annotation rows with columns ``dataset``, ``slide``,
        ``patient``, and ``category``.

    Raises:
        ValueError: If the metadata lacks a slide identifier column or if any
            requested slide id is absent.
    """

    import pandas as pd

    metadata_df = load_tabular_smoke_metadata(gtex_metadata_csv).copy()
    slide_column = _first_present_column(
        metadata_df,
        (
            "slide",
            "slide_id",
            "Tissue Sample Id",
            "file_name",
            "FILE_NAME",
            "filename",
            "image_id",
            "wsi_id",
            "wsi",
            "path",
        ),
    )
    if slide_column is None:
        raise ValueError(
            "GTEx smoke metadata must include a slide identifier column. "
            f"Available columns: {sorted(metadata_df.columns)}"
        )

    metadata_df = metadata_df.assign(
        _normalized_slide_id=metadata_df[slide_column].map(_normalized_slide_id)
    )
    selected = metadata_df.loc[
        metadata_df["_normalized_slide_id"].isin(slide_ids)
    ].copy()
    missing = sorted(set(slide_ids) - set(selected["_normalized_slide_id"].tolist()))
    if missing:
        if strict:
            raise ValueError(
                f"GTEx smoke metadata is missing requested slides: {missing}."
            )
        return pd.DataFrame(
            {
                "dataset": ["hf_gtex_artery"] * len(slide_ids),
                "slide": slide_ids,
                "patient": slide_ids,
                "category": [_default_gtex_category(metadata_df)] * len(slide_ids),
                "age_bracket": ["unknown"] * len(slide_ids),
            }
        )

    patient_column = _first_present_column(
        selected,
        ("patient", "patient_id", "PATIENT_ID", "subject_id", "donor_id"),
    )
    category_column = _first_present_column(
        selected,
        (
            "tissue",
            "tissue_type",
            "sample_type",
            "sample_type_name",
            "Pathology Categories",
            "primary_site",
            "site",
            "category",
        ),
    )

    age_bracket_column = _first_present_column(
        selected,
        ("age_bracket", "Age Bracket", "age_group", "age"),
    )

    annotations = pd.DataFrame(
        {
            "dataset": "hf_gtex_artery",
            "slide": selected["_normalized_slide_id"].astype(str),
            "patient": (
                selected[patient_column].astype(str)
                if patient_column is not None
                else selected["_normalized_slide_id"].astype(str)
            ),
            "category": (
                selected[category_column].astype(str)
                if category_column is not None
                else "artery"
            ),
            "age_bracket": (
                selected[age_bracket_column].astype(str)
                if age_bracket_column is not None
                else "unknown"
            ),
        }
    )
    return annotations.drop_duplicates(subset=["slide"]).reset_index(drop=True)


def merge_survival_metadata(
    feature_observations: Any,
    survival_metadata_csv: Path,
) -> Any:
    """Join feature rows against the full TCGA READ survival metadata table.

    Args:
        feature_observations: ``adata.obs``-like table containing one row per
            feature vector and a slide filename column.
        survival_metadata_csv: Full TCGA READ survival CSV path.

    Returns:
        pd.DataFrame: Joined rows ordered like the feature matrix, including a
        ``feature_row_index`` column and the full survival metadata columns.

    Raises:
        ValueError: If the join columns are unavailable or no rows match.
    """

    obs_df = feature_observations.copy().reset_index(drop=True)
    survival_df = load_tabular_smoke_metadata(survival_metadata_csv).copy()
    obs_slide_column = _first_present_column(
        obs_df, ("FILE_NAME", "file_name", "slide")
    )
    survival_slide_column = _first_present_column(
        survival_df,
        ("FILE_NAME", "file_name", "slide", "slide_id"),
    )
    if obs_slide_column is None or survival_slide_column is None:
        raise ValueError("Survival smoke metadata requires a slide filename column.")

    obs_df = obs_df.assign(
        feature_row_index=np.arange(len(obs_df), dtype=np.int64),
        slide_id=obs_df[obs_slide_column].map(_normalized_slide_id),
    )
    survival_df = survival_df.assign(
        slide_id=survival_df[survival_slide_column].map(_normalized_slide_id)
    )
    merged = obs_df.loc[:, ["feature_row_index", "slide_id"]].merge(
        survival_df.drop_duplicates(subset=["slide_id"]),
        on="slide_id",
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No feature rows matched the TCGA READ survival metadata.")
    return merged.sort_values("feature_row_index").reset_index(drop=True)


def _normalized_slide_id(value: Any) -> str:
    text = Path(str(value)).name
    return text.replace(".gz", "").split(".", 1)[0]


def _first_present_column(frame: Any, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def _default_gtex_category(metadata_df: Any) -> str:
    category_column = _first_present_column(
        metadata_df,
        (
            "tissue",
            "tissue_type",
            "sample_type",
            "sample_type_name",
            "Pathology Categories",
            "primary_site",
            "site",
            "category",
        ),
    )
    if category_column is None:
        return "artery"
    non_null = metadata_df[category_column].dropna().astype(str)
    if non_null.empty:
        return "artery"
    return non_null.mode().iloc[0]


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
        artifact_path: PathForge slide artifact path.
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
    slide-level representation from PathForge tile artifacts without rerunning
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


def describe_output_path(path: Path) -> dict[str, Any]:
    """Describe one smoke output path for persisted investigation reports.

    Args:
        path: File or directory path produced by a smoke workflow.

    Returns:
        dict[str, Any]: JSON-serializable metadata describing the path,
        including existence, kind, and a small amount of file-system context.
    """

    resolved = path.expanduser()
    payload: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.exists(),
    }
    if not resolved.exists():
        payload["kind"] = "missing"
        return payload

    stat_result = resolved.stat()
    payload["size_bytes"] = int(stat_result.st_size)
    if resolved.is_dir():
        payload["kind"] = "directory"
        payload["num_entries"] = sum(1 for _ in resolved.iterdir())
    else:
        payload["kind"] = "file"
        payload["suffix"] = resolved.suffix
    return payload


def smoke_step_artifacts_dir(step_name: str) -> Path | None:
    """Return the per-step artifact directory inside the configured report root.

    Args:
        step_name: Human-readable smoke step name.

    Returns:
        Path | None: Destination directory under ``artifacts/`` when
        ``PATHFORGE_SMOKE_REPORT_DIR`` is configured, otherwise ``None``.
    """

    report_dir = configured_smoke_report_dir()
    if report_dir is None:
        return None
    return report_dir / "artifacts" / step_name.replace(" ", "_").replace("/", "_")


def _mirror_output_path(src: Path, dst: Path) -> Path:
    """Persist one smoke artifact under the report directory.

    Files are hardlinked, symlinked, or copied via ``link_or_copy``. Directory
    trees are copied recursively so the report remains inspectable after pytest
    cleans its temporary workspace.

    Args:
        src: Existing source file or directory.
        dst: Target path inside the smoke report directory.

    Returns:
        Path: The persisted destination path.
    """

    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return dst
    if dst.exists():
        dst.unlink()
    return link_or_copy(src, dst)


def _prepare_output_descriptors(
    outputs: dict[str, Path],
    *,
    step_name: str,
    category: str,
) -> dict[str, dict[str, Any]]:
    """Describe and optionally mirror smoke outputs into the report tree.

    Args:
        outputs: Named smoke outputs to attach to the metrics payload.
        step_name: Human-readable smoke step name.
        category: Output category, for example ``"intermediate"`` or
            ``"final"``.

    Returns:
        dict[str, dict[str, Any]]: JSON-serializable output descriptors.
    """

    artifacts_dir = smoke_step_artifacts_dir(step_name)
    descriptors: dict[str, dict[str, Any]] = {}
    for name, original_path in outputs.items():
        resolved = original_path.expanduser()
        described_path = resolved
        descriptor = describe_output_path(resolved)
        if artifacts_dir is not None and resolved.exists():
            destination_root = artifacts_dir / category / name
            if resolved.is_dir():
                mirrored_path = _mirror_output_path(resolved, destination_root)
            else:
                mirrored_path = _mirror_output_path(
                    resolved,
                    destination_root / resolved.name,
                )
            described_path = mirrored_path
            descriptor = describe_output_path(mirrored_path)
            descriptor["source_path"] = str(resolved)
        descriptors[name] = descriptor
        descriptors[name]["path"] = str(described_path)
    return descriptors


def attach_smoke_outputs(
    payload: dict[str, Any],
    *,
    step_name: str,
    intermediate: dict[str, Path] | None = None,
    final: dict[str, Path] | None = None,
) -> None:
    """Attach intermediate and final output descriptors to one smoke payload.

    Args:
        payload: Mutable metrics payload returned by ``capture_smoke_metrics``.
        step_name: Human-readable smoke step name used for report mirroring.
        intermediate: Named intermediate artifacts or directories.
        final: Named final artifacts or directories.
    """

    if intermediate:
        payload["intermediate_outputs"] = _prepare_output_descriptors(
            intermediate,
            step_name=step_name,
            category="intermediate",
        )
    if final:
        payload["final_outputs"] = _prepare_output_descriptors(
            final,
            step_name=step_name,
            category="final",
        )


def _ru_maxrss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _write_report_step_copy(*, step_name: str, payload: dict[str, Any]) -> None:
    report_dir = configured_smoke_report_dir()
    if report_dir is None:
        return
    steps_dir = report_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    step_path = steps_dir / smoke_metrics_filename(step_name)
    step_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_smoke_report(report_dir: Path) -> dict[str, Path]:
    """Aggregate per-step smoke metrics into JSON and Markdown reports.

    Args:
        report_dir: Root directory containing a ``steps/`` folder with copied
            smoke metrics JSON files.

    Returns:
        dict[str, Path]: Paths to the generated ``json`` and ``markdown``
        reports.
    """

    steps_dir = report_dir / "steps"
    step_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(steps_dir.glob("*.metrics.json"))
    ]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_steps": len(step_payloads),
        "step_names": [payload["step_name"] for payload in step_payloads],
        "steps": step_payloads,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "smoke_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# PathForge Smoke Report",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Steps: `{summary['num_steps']}`",
        "",
    ]
    for payload in step_payloads:
        lines.append(f"## {payload['step_name']}")
        lines.append(f"- Elapsed seconds: `{payload.get('elapsed_seconds', 'n/a')}`")
        lines.append(f"- Peak RSS MB: `{payload.get('ru_maxrss_mb', 'n/a')}`")
        if payload.get("intermediate_outputs"):
            lines.append("- Intermediate outputs:")
            for name, entry in payload["intermediate_outputs"].items():
                lines.append(f"  - `{name}`: `{entry['path']}` ({entry['kind']})")
        if payload.get("final_outputs"):
            lines.append("- Final outputs:")
            for name, entry in payload["final_outputs"].items():
                lines.append(f"  - `{name}`: `{entry['path']}` ({entry['kind']})")
        lines.append("")

    markdown_path = report_dir / "smoke_summary.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


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
        _write_report_step_copy(step_name=step_name, payload=payload)
