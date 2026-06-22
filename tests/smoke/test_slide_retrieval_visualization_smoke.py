"""Smoke tests for slide retrieval visualization rendering.

Exercises render_retrieval_results_image() end-to-end with real slide thumbnails
extracted from the session-scoped GTEx workspace.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from pathbench.core.io.slide_artifacts.base import FileHandleH5
from pathbench.core.io.slide_artifacts import thumbnail as thumbnail_io
from pathbench.slide_retrieval.visualization.renderers import (
    render_retrieval_results_image,
    RESULT_THUMB_SIZE,
)
from ._smoke_dataset import ExtractedWsiWorkspace, attach_smoke_outputs, capture_smoke_metrics


def _load_thumbnail(artifact_path: Path) -> Image.Image:
    """Load and resize a real slide thumbnail from an H5 artifact."""
    with FileHandleH5(artifact_path, mode="r") as slide_artifact:
        image_bytes = thumbnail_io.read_thumbnail_image(slide_artifact)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return img.resize(RESULT_THUMB_SIZE, Image.LANCZOS)


@pytest.mark.smoke
def test_smoke_retrieval_results_visualization_produces_valid_png(
    tmp_path: Path,
    extracted_wsi_workspace: ExtractedWsiWorkspace,
) -> None:
    """render_retrieval_results_image() must write a valid, non-empty PNG for 1 query + 5 hits."""
    artifact_paths = list(sorted(extracted_wsi_workspace.artifact_paths.values()))
    assert len(artifact_paths) >= 6, "Need at least 6 slides for this test"

    slide_ids = list(sorted(extracted_wsi_workspace.artifact_paths.keys()))
    query_id = slide_ids[0]
    ref_ids = slide_ids[1:6]

    query_thumbnail = _load_thumbnail(artifact_paths[0])
    query_lines = [f"slide: {query_id}", "dataset: GTEx"]

    hit_panels = [
        (
            _load_thumbnail(extracted_wsi_workspace.artifact_paths[ref_id]),
            [f"slide: {ref_id}", f"score: {round(0.9 - i * 0.08, 2)}", "dataset: GTEx"],
        )
        for i, ref_id in enumerate(ref_ids)
    ]

    output_dir = tmp_path / "retrieval_visualizations" / "retrieval_results"
    output_dir.mkdir(parents=True)
    output_path = output_dir / f"{query_id}.png"

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="smoke_retrieval_results_visualization",
        metadata={"num_hits": len(hit_panels), "query_id": query_id},
    ) as metadata:
        rendered = render_retrieval_results_image(
            query_thumbnail=query_thumbnail,
            query_lines=query_lines,
            hit_panels=hit_panels,
        )
        rendered.save(output_path, format="PNG")

        attach_smoke_outputs(
            metadata,
            step_name="smoke_retrieval_results_visualization",
            final={"retrieval_results_png": output_path},
        )

    assert output_path.is_file(), "Retrieval results PNG was not written"
    loaded = Image.open(output_path)
    assert loaded.format == "PNG"
    assert loaded.width > RESULT_THUMB_SIZE[0], "Canvas must be wider than a single thumbnail"
    assert loaded.height > RESULT_THUMB_SIZE[1], "Canvas must be taller than a single thumbnail"


@pytest.mark.smoke
def test_smoke_retrieval_results_visualization_multi_query(
    tmp_path: Path,
    extracted_wsi_workspace: ExtractedWsiWorkspace,
) -> None:
    """One PNG per query must be written when multiple queries are rendered."""
    slide_ids = list(sorted(extracted_wsi_workspace.artifact_paths.keys()))
    assert len(slide_ids) >= 10, "Need at least 10 slides for this test"

    query_ids = slide_ids[:5]
    ref_ids = slide_ids[5:10]

    output_dir = tmp_path / "retrieval_visualizations" / "retrieval_results"
    output_dir.mkdir(parents=True)

    written_paths: list[Path] = []

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="smoke_retrieval_results_visualization_multi_query",
        metadata={"num_queries": len(query_ids), "num_refs_per_query": len(ref_ids)},
    ) as metadata:
        for query_id in query_ids:
            query_thumbnail = _load_thumbnail(extracted_wsi_workspace.artifact_paths[query_id])
            hit_panels = [
                (
                    _load_thumbnail(extracted_wsi_workspace.artifact_paths[ref_id]),
                    [f"slide: {ref_id}", f"score: {round(0.9 - i * 0.05, 2)}"],
                )
                for i, ref_id in enumerate(ref_ids)
            ]
            rendered = render_retrieval_results_image(
                query_thumbnail=query_thumbnail,
                query_lines=[f"slide: {query_id}"],
                hit_panels=hit_panels,
            )
            out = output_dir / f"{query_id}.png"
            rendered.save(out, format="PNG")
            written_paths.append(out)

        attach_smoke_outputs(
            metadata,
            step_name="smoke_retrieval_results_visualization_multi_query",
            final={f"png_{i}": p for i, p in enumerate(written_paths)},
        )

    assert len(written_paths) == len(query_ids)
    for path in written_paths:
        assert path.is_file(), f"Missing PNG: {path}"
        img = Image.open(path)
        assert img.format == "PNG"
        assert img.width > 0 and img.height > 0
