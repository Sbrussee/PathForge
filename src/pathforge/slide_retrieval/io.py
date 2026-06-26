from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from pathforge.core.io.slide_artifacts import tiles as tiles_io
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.core.io.slide_retrieval import (
    retrieval_representations as retrieval_representations_io,
)
from pathforge.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathforge.slide_retrieval.search_strategies.types import SearchResult
from pathforge.slide_retrieval.search_strategies.types import SearchHit
from pathforge.slide_retrieval.types import (
    RetrievalItemIdentity,
    SlideRetrievalManifest,
)

_RANK_SAMPLE_PATTERN = re.compile(r"^rank_(?P<rank>[1-9]\d*)_sample_id$")


def _safe_output_name(value: Any, *, hyphenate_underscores: bool = False) -> str:
    text = str(value).strip().lower()
    if hyphenate_underscores:
        text = text.replace("_", "-")
    text = text.replace(" ", "-")
    text = text.replace("/", "-")
    text = text.replace("\\", "-")
    while "--" in text:
        text = text.replace("--", "-")

    allowed: list[str] = []
    for character in text:
        if character.isalnum() or character in "._-":
            allowed.append(character)
        else:
            allowed.append("-")
    return "".join(allowed).strip("-")


def _safe_inference_output_name(value: Any) -> str:
    text = str(value).strip().lower()
    for old, new in (
        (" ", "_"),
        ("/", "_"),
        ("\\", "_"),
        ("-", "_"),
    ):
        text = text.replace(old, new)

    allowed: list[str] = []
    for character in text:
        if character.isalnum() or character in "._":
            allowed.append(character)
        else:
            allowed.append("_")

    safe_name = "".join(allowed)
    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")
    return safe_name.strip("_")


def build_slide_retrieval_inference_output_root(
    *,
    inference_run_root: str | Path,
    tiling_id: str,
    feature_name: str,
    slide_representation: str,
    search_method: str,
    run_hash: str,
) -> Path:
    """
    Build the flat inference output root for one slide-retrieval combo.

    Layout:
    - `{inference_run_root}/{tiling_id}_{feature_name}/`
      `{slide_representation}_{search_method}_{run_hash12}/`
    """
    tiling_component = (
        f"{_safe_inference_output_name(tiling_id)}_"
        f"{_safe_inference_output_name(feature_name)}"
    )
    method_component = (
        f"{_safe_inference_output_name(slide_representation)}_"
        f"{_safe_inference_output_name(search_method)}_"
        f"{_safe_inference_output_name(str(run_hash)[:12])}"
    )
    return Path(inference_run_root) / tiling_component / method_component


def build_slide_retrieval_output_root(
    *,
    project_root: str,
    tiling_id: str,
    feature_name: str,
    slide_representation: str,
    search_method: str,
) -> Path:
    """
    Build the canonical search-specific output root for one slide-retrieval combo.

    Inputs:
    - `project_root`: experiment project root.
    - `tiling_id`: canonical tiling identifier.
    - `feature_name`: canonical stored feature name.
    - `slide_representation`: retrieval representation strategy name.
    - `search_method`: search strategy name.

    Returns:
    - `Path` to the search-specific slide-retrieval output root.
    """

    representation_root = build_slide_retrieval_representation_root(
        project_root=project_root,
        tiling_id=tiling_id,
        feature_name=feature_name,
        slide_representation=slide_representation,
    )
    search_component = _safe_output_name(search_method, hyphenate_underscores=True)
    return representation_root / search_component


def build_slide_retrieval_representation_root(
    *,
    project_root: str,
    tiling_id: str,
    feature_name: str,
    slide_representation: str,
) -> Path:
    """
    Build the canonical representation-specific root for slide-retrieval outputs.

    Layout:
    - `eval_slide_retrieval/<tiling+feature>/<retrieval_representation>/`
    """

    tiling_component = f"{_safe_output_name(tiling_id)}_{_safe_output_name(feature_name)}"
    representation_component = _safe_output_name(
        slide_representation,
        hyphenate_underscores=True,
    )
    return (
        Path(project_root)
        / "eval_slide_retrieval"
        / tiling_component
        / representation_component
    )


def load_sample_patch_coords(
    *,
    sample: Any,
    tile_id: str,
    dtype: np.dtype | type[np.integer[Any]] = np.int32,
) -> np.ndarray:
    """Load patch coordinates for one sample from its retrieval artifact."""
    if sample is None:
        raise ValueError("sample is required to load patch coordinates.")

    artifact_paths = list(getattr(sample, "artifact_paths", []) or [])
    if not artifact_paths:
        raise ValueError("sample.artifact_paths is required to load patch coordinates.")

    coords_parts: list[np.ndarray] = []
    for artifact_path in artifact_paths:
        with FileHandleH5(artifact_path, mode="r") as slide_artifact:
            slide_coords = tiles_io.read_coords(slide_artifact, bag_id=tile_id)
        coords_parts.append(np.asarray(slide_coords[:, :2], dtype=dtype))

    if not coords_parts:
        return np.empty((0, 2), dtype=dtype)

    return np.concatenate(coords_parts, axis=0).astype(dtype, copy=False)


def load_slide_retrieval_representation(
    *,
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    representation_id: str,
    entry_id: str | None,
) -> RetrievalRepresentation | None:
    """Load a stored slide-retrieval representation from disk."""
    if not retrieval_representations_io.retrieval_representation_entry_exists(
        retrieval_artifact=retrieval_artifact,
        tile_id=tile_id,
        representation_id=representation_id,
        entry_id=entry_id,
    ):
        return None

    entry = retrieval_representations_io.read_retrieval_representation_entry(
        retrieval_artifact=retrieval_artifact,
        tile_id=tile_id,
        representation_id=representation_id,
        entry_id=entry_id,
    )
    identity = RetrievalItemIdentity.from_dict(
        entry["metadata"],
        fallback_sample_id=entry_id,
    )
    return RetrievalRepresentation(
        sample_id=identity.sample_id,
        exclusion_key=identity.exclusion_key,
        data=entry["embedding"],
        additional_data=dict(entry["additional_data"]),
    )


def save_slide_retrieval_representation(
    *,
    retrieval_artifact: FileHandleH5,
    tile_id: str,
    representation_id: str,
    entry_id: str | None,
    representation: RetrievalRepresentation,
    params: dict[str, Any] | None = None,
) -> None:
    """Persist a slide-retrieval representation to disk."""
    retrieval_representations_io.write_retrieval_representation_entry(
        retrieval_artifact=retrieval_artifact,
        tile_id=tile_id,
        representation_id=representation_id,
        entry_id=entry_id,
        metadata=RetrievalItemIdentity(
            sample_id=representation.sample_id,
        ).to_dict(),
        embedding=representation.data,
        params=dict(params or {}),
        additional_data=representation.additional_data,
    )


def write_slide_retrieval_manifest(
    path: str | Path,
    manifest: SlideRetrievalManifest | Mapping[str, Any],
) -> None:
    """Write the slide-retrieval run manifest as JSON."""
    path = Path(path)
    payload = manifest.to_dict() if hasattr(manifest, "to_dict") else dict(manifest)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_slide_retrieval_results_csv(
    path: str | Path,
    results: list[SearchResult],
) -> None:
    """Backward-compatible wrapper that now writes Excel ranked results."""
    write_slide_retrieval_results_xlsx(path, results)


def write_slide_retrieval_results_xlsx(
    path: str | Path,
    results: list[SearchResult],
) -> None:
    """Write ranked slide-retrieval query results to an Excel workbook."""
    path = Path(path)
    if path.suffix.lower() != ".xlsx":
        path = path.with_suffix(".xlsx")
    max_hits = max((len(result.hits) for result in results), default=0)

    fieldnames = ["query_sample_id"]
    for rank in range(1, max_hits + 1):
        fieldnames.extend(
            [
                f"rank_{rank}_sample_id",
                f"rank_{rank}_score",
            ]
        )

    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "query_sample_id": str(result.query_sample_id),
        }

        hits = sorted(result.hits, key=lambda hit: hit.rank)
        for idx, hit in enumerate(hits, start=1):
            row[f"rank_{idx}_sample_id"] = str(hit.sample_id)
            row[f"rank_{idx}_score"] = _stringify(hit.score)

        rows.append(row)

    pd.DataFrame(rows, columns=fieldnames).to_excel(path, index=False)


def resolve_slide_retrieval_results_path(path: str | Path) -> Path:
    """
    Resolve the ranked-results file for a run, preferring Excel over legacy CSV.
    """
    path = Path(path)
    candidates: list[Path]

    if path.suffix.lower() == ".xlsx":
        candidates = [path, path.with_suffix(".csv")]
    elif path.suffix.lower() == ".csv":
        candidates = [path.with_suffix(".xlsx"), path]
    else:
        candidates = [path.with_suffix(".xlsx"), path.with_suffix(".csv"), path]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return candidates[0]


def read_slide_retrieval_results_csv(path: str | Path) -> list[SearchResult]:
    """
    Read one saved slide-retrieval ranked-results CSV.

    Inputs:
    - `path`: CSV path written by `write_slide_retrieval_results_csv`.

    Returns:
    - List of `SearchResult` objects with ranked hits reconstructed from CSV.
    """
    path = resolve_slide_retrieval_results_path(path)
    if path.suffix.lower() == ".xlsx":
        results_df = pd.read_excel(path)
        records = results_df.to_dict(orient="records")
    else:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            records = list(reader)

    results: list[SearchResult] = []
    for row in records:
        hits = []
        for column_name, value in row.items():
            match = _RANK_SAMPLE_PATTERN.fullmatch(str(column_name))
            if match is None or pd.isna(value) or value == "":
                continue

            rank = int(match.group("rank"))
            score_raw = row.get(f"rank_{rank}_score", "")
            score = 0.0 if pd.isna(score_raw) or score_raw in ("", None) else float(score_raw)
            hits.append(
                SearchHit(
                    sample_id=str(value),
                    score=score,
                    rank=rank,
                )
            )

        results.append(
            SearchResult(
                query_sample_id=str(row["query_sample_id"]),
                hits=sorted(hits, key=lambda hit: hit.rank),
            )
        )

    return results


def write_metrics_csv(path: str | Path, metrics: Mapping[str, Mapping[str, Any]]) -> None:
    """
    Write legacy flat metric rows for retrieval metrics.

    Inputs:
    - `path`: destination CSV path.
    - `metrics`: mapping from metric name to JSON-like payload.

    Returns:
    - `None`. Writes one flat row per metric scope.
    """

    path = Path(path)
    fieldnames = ["metric", "scope", "label", "value"]
    rows: list[dict[str, str]] = []

    for metric_name, payload in metrics.items():
        per_label = payload.get("per_label", payload.get("per_class", {}))
        for label, value in per_label.items():
            rows.append(
                {
                    "metric": str(metric_name),
                    "scope": "per_class",
                    "label": str(label),
                    "value": str(value),
                }
            )

        for key in ("macro", "micro"):
            if key not in payload:
                continue
            rows.append(
                {
                    "metric": str(metric_name),
                    "scope": key,
                    "label": "",
                    "value": str(payload[key]),
                }
            )

        for key, value in payload.items():
            if key in {
                "per_label",
                "per_class",
                "counts",
                "counts_per_label",
                "macro",
                "micro",
            }:
                continue
            rows.append(
                {
                    "metric": str(metric_name),
                    "scope": str(key),
                    "label": "",
                    "value": str(value),
                }
            )

        counts = payload.get("counts", {})
        for key, value in counts.items():
            rows.append(
                {
                    "metric": str(metric_name),
                    "scope": str(key),
                    "label": "",
                    "value": str(value),
                }
            )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
