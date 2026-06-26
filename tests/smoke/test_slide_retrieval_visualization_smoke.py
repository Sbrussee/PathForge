"""Smoke tests for slide retrieval visualization rendering.

Renders ``render_retrieval_results_image()`` end-to-end using real slide
thumbnails from the session-scoped GTEx workspace **and the real ranked hits
and distances produced by an actual Yottixel search** over the slides' real
features — so the visualizations show genuine queries and scores, not
hand-fabricated values.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.core.io.slide_artifacts import thumbnail as thumbnail_io
from pathforge.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathforge.slide_retrieval.search_strategies.strategies.yottixel import (
    YottixelSearch,
)
from pathforge.slide_retrieval.visualization.renderers import (
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


def _real_search(
    *,
    query_id: str,
    ref_ids: list[str],
    slide_ids: list[str],
    feature_matrix,
):
    """Run a real Yottixel search (query vs references) on real features.

    Returns the search result whose ranked hits carry the real reference
    ``item_id``s and Yottixel distances (lower = more similar).
    """

    def _rep(sample_id: str) -> RetrievalRepresentation:
        idx = slide_ids.index(sample_id)
        return RetrievalRepresentation(
            sample_id=sample_id,
            data=[feature_matrix[idx].tolist()],
            exclusion_key=None,
        )

    strategy = YottixelSearch(params={"k": len(ref_ids)})
    strategy.build_database([_rep(ref_id) for ref_id in ref_ids])
    return strategy.search(_rep(query_id))


def _hit_panels_from_result(result, workspace: ExtractedWsiWorkspace):
    """Build (thumbnail, lines) panels from real ranked hits and their scores."""
    return [
        (
            _load_thumbnail(workspace.artifact_paths[hit.item_id]),
            [f"slide: {hit.item_id}", f"score: {hit.score:.3f}", "dataset: GTEx"],
        )
        for hit in result.hits
    ]


@pytest.mark.smoke
def test_smoke_retrieval_results_visualization_produces_valid_png(
    tmp_path: Path,
    extracted_wsi_workspace: ExtractedWsiWorkspace,
    slide_level_feature_matrix: tuple,
) -> None:
    """Render a 1-query/5-hit panel from a real Yottixel search over real features."""
    slide_ids, feature_matrix = slide_level_feature_matrix
    ordered_ids = list(sorted(extracted_wsi_workspace.artifact_paths.keys()))
    assert len(ordered_ids) >= 6, "Need at least 6 slides for this test"

    query_id = ordered_ids[0]
    ref_ids = ordered_ids[1:6]

    result = _real_search(
        query_id=query_id,
        ref_ids=ref_ids,
        slide_ids=slide_ids,
        feature_matrix=feature_matrix,
    )

    # The rendered hits/scores are the *real* search output.
    assert result.query_id == query_id
    assert len(result.hits) == len(ref_ids)
    assert all(hit.item_id in ref_ids for hit in result.hits)
    assert [hit.rank for hit in result.hits] == list(range(1, len(result.hits) + 1))
    scores = [hit.score for hit in result.hits]
    assert scores == sorted(scores), "Yottixel hits must be ranked by ascending distance"

    query_thumbnail = _load_thumbnail(extracted_wsi_workspace.artifact_paths[query_id])
    hit_panels = _hit_panels_from_result(result, extracted_wsi_workspace)

    output_dir = tmp_path / "retrieval_visualizations" / "retrieval_results"
    output_dir.mkdir(parents=True)
    output_path = output_dir / f"{query_id}.png"

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="smoke_retrieval_results_visualization",
        metadata={
            "num_hits": len(hit_panels),
            "query_id": query_id,
            "top_hit": result.hits[0].item_id,
            "top_score": float(result.hits[0].score),
            "hit_scores": [float(hit.score) for hit in result.hits],
        },
    ) as metadata:
        rendered = render_retrieval_results_image(
            query_thumbnail=query_thumbnail,
            query_lines=[f"slide: {query_id}", "dataset: GTEx"],
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
    slide_level_feature_matrix: tuple,
) -> None:
    """One PNG per query, each rendered from that query's real ranked search hits."""
    slide_ids, feature_matrix = slide_level_feature_matrix
    ordered_ids = list(sorted(extracted_wsi_workspace.artifact_paths.keys()))
    assert len(ordered_ids) >= 10, "Need at least 10 slides for this test"

    query_ids = ordered_ids[:5]
    ref_ids = ordered_ids[5:10]

    output_dir = tmp_path / "retrieval_visualizations" / "retrieval_results"
    output_dir.mkdir(parents=True)

    written_paths: list[Path] = []
    per_query_scores: dict[str, list[float]] = {}

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="smoke_retrieval_results_visualization_multi_query",
        metadata={"num_queries": len(query_ids), "num_refs_per_query": len(ref_ids)},
    ) as metadata:
        for query_id in query_ids:
            result = _real_search(
                query_id=query_id,
                ref_ids=ref_ids,
                slide_ids=slide_ids,
                feature_matrix=feature_matrix,
            )
            assert result.query_id == query_id
            assert all(hit.item_id in ref_ids for hit in result.hits)
            scores = [hit.score for hit in result.hits]
            assert scores == sorted(scores)
            per_query_scores[query_id] = [float(s) for s in scores]

            query_thumbnail = _load_thumbnail(
                extracted_wsi_workspace.artifact_paths[query_id]
            )
            hit_panels = _hit_panels_from_result(result, extracted_wsi_workspace)
            rendered = render_retrieval_results_image(
                query_thumbnail=query_thumbnail,
                query_lines=[f"slide: {query_id}"],
                hit_panels=hit_panels,
            )
            out = output_dir / f"{query_id}.png"
            rendered.save(out, format="PNG")
            written_paths.append(out)

        metadata["per_query_hit_scores"] = per_query_scores
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
