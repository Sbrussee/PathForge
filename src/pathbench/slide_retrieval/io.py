from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from pathbench.core.io.slide_artifacts import tiles as tiles_io
from pathbench.core.io.slide_artifacts.base import FileHandleH5
from pathbench.core.io.slide_retrieval import (
    retrieval_representations as retrieval_representations_io,
)
from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathbench.slide_retrieval.search_strategies.types import SearchResult
from pathbench.slide_retrieval.search_strategies.types import SearchHit
from pathbench.slide_retrieval.types import (
    RetrievalItemIdentity,
    SlideRetrievalManifest,
)

_RANK_SAMPLE_PATTERN = re.compile(r"^rank_(?P<rank>[1-9]\d*)_sample_id$")


def _safe_output_name(value: Any, *, hyphenate_underscores: bool = False) -> str:
    text = str(value).strip()
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


def build_slide_retrieval_output_root(
    *,
    project_root: str,
    tiling_id: str,
    feature_name: str,
    slide_representation: str,
    search_method: str,
) -> Path:
    """
    Build the canonical output root for one slide-retrieval combo.

    Inputs:
    - `project_root`: experiment project root.
    - `tiling_id`: canonical tiling identifier.
    - `feature_name`: canonical stored feature name.
    - `slide_representation`: retrieval representation strategy name.
    - `search_method`: search strategy name.

    Returns:
    - `Path` to the combo-specific slide-retrieval output root.
    """

    tiling_component = f"{_safe_output_name(tiling_id)}_{_safe_output_name(feature_name)}"
    method_component = (
        f"{_safe_output_name(slide_representation)}_"
        f"{_safe_output_name(search_method, hyphenate_underscores=True)}"
    )
    return (
        Path(project_root)
        / "eval"
        / "slide_retrieval"
        / tiling_component
        / method_component
    )


def load_sample_patch_coords(
    *,
    sample: Any,
    tile_id: str,
    dtype: np.dtype | type[np.integer[Any]] = np.int32,
) -> np.ndarray:
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
    entry_id: str,
) -> RetrievalRepresentation | None:
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
    entry_id: str,
    representation: RetrievalRepresentation,
    params: dict[str, Any] | None = None,
) -> None:
    retrieval_representations_io.write_retrieval_representation_entry(
        retrieval_artifact=retrieval_artifact,
        tile_id=tile_id,
        representation_id=representation_id,
        entry_id=entry_id,
        metadata=RetrievalItemIdentity(
            sample_id=representation.sample_id,
            exclusion_key=representation.exclusion_key,
        ).to_dict(),
        embedding=representation.data,
        params=dict(params or {}),
        additional_data=representation.additional_data,
    )


def write_slide_retrieval_manifest(
    path: str | Path,
    manifest: SlideRetrievalManifest | Mapping[str, Any],
) -> None:
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
    path = Path(path)
    max_hits = max((len(result.hits) for result in results), default=0)

    fieldnames = ["query_sample_id"]
    for rank in range(1, max_hits + 1):
        fieldnames.extend(
            [
                f"rank_{rank}_sample_id",
                f"rank_{rank}_score",
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row: dict[str, Any] = {
                "query_sample_id": str(result.query_sample_id),
            }

            hits = sorted(result.hits, key=lambda hit: hit.rank)
            for idx, hit in enumerate(hits, start=1):
                row[f"rank_{idx}_sample_id"] = str(hit.sample_id)
                row[f"rank_{idx}_score"] = _stringify(hit.score)

            writer.writerow(row)


def read_slide_retrieval_results_csv(path: str | Path) -> list[SearchResult]:
    """
    Read one saved slide-retrieval ranked-results CSV.

    Inputs:
    - `path`: CSV path written by `write_slide_retrieval_results_csv`.

    Returns:
    - List of `SearchResult` objects with ranked hits reconstructed from CSV.
    """

    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        results: list[SearchResult] = []
        for row in reader:
            hits = []
            for column_name, value in row.items():
                match = _RANK_SAMPLE_PATTERN.fullmatch(str(column_name))
                if match is None or not value:
                    continue

                rank = int(match.group("rank"))
                score_raw = row.get(f"rank_{rank}_score", "")
                score = float(score_raw) if score_raw not in ("", None) else 0.0
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
