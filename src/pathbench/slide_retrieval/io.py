from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from pathbench.core.io.h5 import tiles as tiles_io
from pathbench.core.io.h5 import retrieval_representations as retrieval_representations_io
from pathbench.core.io.h5.base import FileHandleH5
from pathbench.slide_retrieval.search_strategies.types import (
    SearchDatabaseItem,
    SearchResult,
)
from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathbench.slide_retrieval.types import (
    RetrievalItemMetadata,
    SlideRetrievalManifest,
    SlideRetrievalOutputPaths,
    SlideRetrievalRunSpec,
)


def load_sample_patch_coords(
    *,
    sample: Any,
    bag_id: str,
    dtype: np.dtype | type[np.integer[Any]] = np.int32,
) -> np.ndarray:
    """
    Load row-aligned patch coordinates for one retrieval sample.

    Inputs:
        sample:
            Sample-like object exposing ``artifact_paths`` aligned with the bag
            rows.
        bag_id:
            Canonical tiling identifier used in the slide H5 layout.
        dtype:
            Integer dtype used for the returned ``(x, y)`` coordinate matrix.

    Outputs:
        Returns ``np.ndarray`` with shape ``(N, 2)`` containing ``(x, y)``
        patch coordinates in bag-row order.

    Example:
        ```python
        coords = load_sample_patch_coords(sample=sample, bag_id="256px_0.5mpp")
        ```
    """
    if sample is None:
        raise ValueError("sample is required to load patch coordinates.")

    artifact_paths = list(getattr(sample, "artifact_paths", []) or [])
    if not artifact_paths:
        raise ValueError("sample.artifact_paths is required to load patch coordinates.")

    coords_parts: list[np.ndarray] = []
    for artifact_path in artifact_paths:
        with FileHandleH5(artifact_path, mode="r") as slide_artifact:
            slide_coords = tiles_io.read_coords(slide_artifact, bag_id=bag_id)
        coords_parts.append(np.asarray(slide_coords[:, :2], dtype=dtype))

    if not coords_parts:
        return np.empty((0, 2), dtype=dtype)

    return np.concatenate(coords_parts, axis=0).astype(dtype, copy=False)


def write_slide_retrieval_eval_outputs(
    *,
    run_spec: SlideRetrievalRunSpec,
    results: list[SearchResult],
    reference_items: list[SearchDatabaseItem],
    metrics: dict[str, Any] | None = None,
) -> Path:
    """
    Write all human-readable eval outputs for one image retrieval run.

    Output structure:
        eval/
          slide_retrieval/
            <bag_id>_<feature_extraction>/
              <slide_representation>_<search_method>/
                run_<hash>/
                  manifest.json
                  metrics.csv
                  query_results.csv
                  aggregation_membership.csv
    """
    output_paths = SlideRetrievalOutputPaths.from_run_spec(run_spec)
    output_paths.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_slide_retrieval_manifest(
        run_spec=run_spec,
        results=results,
        reference_items=reference_items,
    )

    write_manifest(output_paths.manifest_path, manifest)
    write_metrics_csv(output_paths.metrics_csv_path, metrics or {})
    write_query_results_csv(output_paths.query_results_csv_path, results)
    write_aggregation_membership_csv(
        output_paths.aggregation_membership_csv_path,
        results=results,
        reference_items=reference_items,
    )

    return output_paths.output_dir


def load_slide_retrieval_representation(
    *,
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    entry_id: str,
) -> RetrievalRepresentation | None:
    """Load one retrieval representation from the shared H5 storage schema."""
    if not retrieval_representations_io.retrieval_representation_item_exists(
        slide_artifact=slide_artifact,
        bag_id=bag_id,
        representation_id=representation_id,
        entry_id=entry_id,
    ):
        return None

    additional_group_path = (
        retrieval_representations_io.DEFAULT_LAYOUT.retrieval_representation_additional_data_group(
            bag_id,
            representation_id,
            entry_id,
        )
    )
    additional_data: dict[str, Any] = {}
    if additional_group_path in slide_artifact.h5:
        for name in slide_artifact.h5[additional_group_path].keys():
            additional_data[name] = (
                retrieval_representations_io.read_additional_retrieval_representation_data(
                    slide_artifact=slide_artifact,
                    bag_id=bag_id,
                    representation_id=representation_id,
                    entry_id=entry_id,
                    name=name,
                )
            )

    return RetrievalRepresentation(
        sample_id=retrieval_representations_io.read_retrieval_representation_sample_id(
            slide_artifact=slide_artifact,
            bag_id=bag_id,
            representation_id=representation_id,
            entry_id=entry_id,
        ),
        representation_type=retrieval_representations_io.read_retrieval_representation_type(
            slide_artifact=slide_artifact,
            bag_id=bag_id,
            representation_id=representation_id,
            entry_id=entry_id,
        ),
        data=retrieval_representations_io.read_retrieval_representation(
            slide_artifact=slide_artifact,
            bag_id=bag_id,
            representation_id=representation_id,
            entry_id=entry_id,
        ),
        metadata=RetrievalItemMetadata.from_dict(
            retrieval_representations_io.read_retrieval_representation_metadata(
                slide_artifact=slide_artifact,
                bag_id=bag_id,
                representation_id=representation_id,
                entry_id=entry_id,
            )
        ),
        additional_data=additional_data,
    )


def save_slide_retrieval_representation(
    *,
    slide_artifact: FileHandleH5,
    bag_id: str,
    representation_id: str,
    entry_id: str,
    representation: RetrievalRepresentation,
    params: dict[str, Any] | None = None,
    slide_ids: list[str] | None = None,
) -> None:
    """Persist one retrieval representation using the shared H5 storage schema."""
    retrieval_representations_io.write_retrieval_representation_sample_id(
        slide_artifact=slide_artifact,
        bag_id=bag_id,
        representation_id=representation_id,
        entry_id=entry_id,
        sample_id=representation.sample_id,
    )
    retrieval_representations_io.write_retrieval_representation_type(
        slide_artifact=slide_artifact,
        bag_id=bag_id,
        representation_id=representation_id,
        entry_id=entry_id,
        representation_type=representation.representation_type,
    )
    retrieval_representations_io.write_retrieval_representation(
        slide_artifact=slide_artifact,
        bag_id=bag_id,
        representation_id=representation_id,
        entry_id=entry_id,
        data=representation.data,
    )
    retrieval_representations_io.write_retrieval_representation_metadata(
        slide_artifact=slide_artifact,
        bag_id=bag_id,
        representation_id=representation_id,
        entry_id=entry_id,
        metadata=representation.metadata.to_dict(),
    )
    retrieval_representations_io.write_retrieval_representation_params(
        slide_artifact=slide_artifact,
        bag_id=bag_id,
        representation_id=representation_id,
        entry_id=entry_id,
        params=dict(params or {}),
    )
    retrieval_representations_io.write_retrieval_representation_slide_ids(
        slide_artifact=slide_artifact,
        bag_id=bag_id,
        representation_id=representation_id,
        entry_id=entry_id,
        slide_ids=list(slide_ids or []),
    )

    additional_group_path = (
        retrieval_representations_io.DEFAULT_LAYOUT.retrieval_representation_additional_data_group(
            bag_id,
            representation_id,
            entry_id,
        )
    )
    if additional_group_path in slide_artifact.h5:
        del slide_artifact.h5[additional_group_path]

    for name, value in representation.additional_data.items():
        retrieval_representations_io.write_additional_retrieval_representation_data(
            slide_artifact=slide_artifact,
            bag_id=bag_id,
            representation_id=representation_id,
            entry_id=entry_id,
            name=name,
            data=value,
        )


def build_slide_retrieval_manifest(
    *,
    run_spec: SlideRetrievalRunSpec,
    results: list[SearchResult],
    reference_items: list[SearchDatabaseItem],
) -> SlideRetrievalManifest:
    """
    Build the manifest payload for one eval run.
    """
    top_k_saved = max((len(result.hits) for result in results), default=0)

    return SlideRetrievalManifest(
        run_spec=run_spec,
        num_queries=len(results),
        num_reference_items=len(reference_items),
        top_k_saved=top_k_saved,
    )


def write_manifest(
    path: str | Path,
    manifest: SlideRetrievalManifest,
) -> None:
    path = Path(path)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_query_results_csv(
    path: str | Path,
    results: list[SearchResult],
) -> None:
    """
    Write the main retrieval overview.

    One row per query:
    - query_id
    - query_category
    - rank_1_id / score / category
    - rank_2_id / score / category
    - ...
    """
    path = Path(path)
    max_hits = max((len(result.hits) for result in results), default=0)

    fieldnames = ["query_id", "query_category"]
    for rank in range(1, max_hits + 1):
        fieldnames.extend(
            [
                f"rank_{rank}_id",
                f"rank_{rank}_score",
                f"rank_{rank}_category",
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            query_meta = result.metadata

            row: dict[str, Any] = {
                "query_id": str(result.query_id),
                "query_category": _stringify(query_meta.category),
            }

            hits = sorted(result.hits, key=lambda hit: hit.rank)

            for idx, hit in enumerate(hits, start=1):
                hit_meta = hit.metadata

                row[f"rank_{idx}_id"] = str(hit.item_id)
                row[f"rank_{idx}_score"] = _stringify(hit.score)
                row[f"rank_{idx}_category"] = _stringify(hit_meta.category)

            writer.writerow(row)


def write_metrics_csv(
    path: str | Path,
    metrics: dict[str, Any],
) -> None:
    """
    Write flat retrieval metrics to CSV.

    Inputs:
    - `path`: `str | Path` output path for `metrics.csv`.
    - `metrics`: `dict[str, Any]` metric payload produced by the retrieval task.
      The expected shape is one top-level key per metric, where each value is a
      mapping that may contain scalar fields plus a `per_class` mapping.

    Returns:
    - `None`. Writes one row per metric aggregate or per-class score.

    Example:
        ```python
        write_metrics_csv(
            "metrics.csv",
            {"hit_at_5": {"macro": 0.7, "micro": 0.8, "per_class": {"tumor": 1.0}}},
        )
        ```
    """

    path = Path(path)
    fieldnames = ["metric", "scope", "label", "value"]
    rows: list[dict[str, str]] = []

    for metric_name, payload in metrics.items():
        if not isinstance(payload, dict):
            rows.append(
                {
                    "metric": str(metric_name),
                    "scope": "metric",
                    "label": "",
                    "value": _stringify(payload),
                }
            )
            continue

        per_class = payload.get("per_class", {})
        if isinstance(per_class, dict):
            for label, value in per_class.items():
                rows.append(
                    {
                        "metric": str(metric_name),
                        "scope": "per_class",
                        "label": _stringify(label),
                        "value": _stringify(value),
                    }
                )

        for aggregate_name in ("macro", "micro"):
            if aggregate_name in payload:
                rows.append(
                    {
                        "metric": str(metric_name),
                        "scope": aggregate_name,
                        "label": "",
                        "value": _stringify(payload[aggregate_name]),
                    }
                )

        for meta_name in ("k", "num_queries", "insufficient_k_queries"):
            if meta_name in payload:
                rows.append(
                    {
                        "metric": str(metric_name),
                        "scope": meta_name,
                        "label": "",
                        "value": _stringify(payload[meta_name]),
                    }
                )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_aggregation_membership_csv(
    path: str | Path,
    *,
    results: list[SearchResult],
    reference_items: list[SearchDatabaseItem],
) -> None:
    """
    Write one wide lookup table for aggregated query/reference items.

    Columns:
    - use
    - aggregated_id
    - member_1
    - member_2
    - ...
    """
    path = Path(path)

    rows: list[dict[str, Any]] = []
    max_members = 0

    seen_query_ids: set[str] = set()
    for result in results:
        aggregated_id = str(result.query_id)
        if aggregated_id in seen_query_ids:
            continue

        seen_query_ids.add(aggregated_id)
        metadata = result.metadata
        members = list(metadata.member_ids)

        max_members = max(max_members, len(members))
        rows.append(
            {
                "use": "query",
                "aggregated_id": aggregated_id,
                "members": members,
            }
        )

    seen_reference_ids: set[str] = set()
    for item in reference_items:
        aggregated_id = str(item.item_id)
        if aggregated_id in seen_reference_ids:
            continue

        seen_reference_ids.add(aggregated_id)
        metadata = item.metadata
        members = list(metadata.member_ids)

        max_members = max(max_members, len(members))
        rows.append(
            {
                "use": "reference",
                "aggregated_id": aggregated_id,
                "members": members,
            }
        )

    fieldnames = ["use", "aggregated_id"] + [
        f"member_{idx}" for idx in range(1, max_members + 1)
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            out_row = {
                "use": row["use"],
                "aggregated_id": row["aggregated_id"],
            }
            for idx, member_id in enumerate(row["members"], start=1):
                out_row[f"member_{idx}"] = member_id

            writer.writerow(out_row)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
