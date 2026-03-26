from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pathbench.benchmarking.tasks.slide_retrieval import SlideRetrievalTask
from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathbench.slide_retrieval.search_strategies.types import SearchHit, SearchResult
from pathbench.slide_retrieval.types import RetrievalItemMetadata


def _make_task(tmp_path: Path, *, evaluation: list[str]) -> SlideRetrievalTask:
    cfg = SimpleNamespace(experiment=SimpleNamespace(evaluation=evaluation))
    experiment = SimpleNamespace(cfg=cfg, project_root=str(tmp_path))
    return SlideRetrievalTask(experiment)


def test_evaluate_results_runs_registered_hit_metric(tmp_path: Path) -> None:
    task = _make_task(tmp_path, evaluation=["hit_at_1"])
    query_representations = [
        RetrievalRepresentation(
            sample_id="query-1",
            representation_type="single_vector",
            data=[1.0],
            metadata=RetrievalItemMetadata(category="tumor", patient_id="patient-q1"),
        ),
        RetrievalRepresentation(
            sample_id="query-2",
            representation_type="single_vector",
            data=[2.0],
            metadata=RetrievalItemMetadata(category="normal", patient_id="patient-q2"),
        ),
    ]
    results = [
        SearchResult(
            query_id="query-1",
            metadata=RetrievalItemMetadata(category="tumor", patient_id="patient-q1"),
            hits=[
                SearchHit(
                    item_id="ref-1",
                    score=0.99,
                    rank=1,
                    metadata=RetrievalItemMetadata(
                        category="tumor",
                        patient_id="patient-r1",
                    ),
                )
            ],
        ),
        SearchResult(
            query_id="query-2",
            metadata=RetrievalItemMetadata(category="normal", patient_id="patient-q2"),
            hits=[
                SearchHit(
                    item_id="ref-2",
                    score=0.85,
                    rank=1,
                    metadata=RetrievalItemMetadata(
                        category="tumor",
                        patient_id="patient-r2",
                    ),
                )
            ],
        ),
    ]

    metrics = task._evaluate_results(
        results=results,
        query_representations=query_representations,
    )

    assert metrics["hit_at_1"]["per_class"] == {"tumor": 1.0, "normal": 0.0}
    assert metrics["hit_at_1"]["macro"] == pytest.approx(0.5)
    assert metrics["hit_at_1"]["micro"] == pytest.approx(0.5)


def test_evaluate_results_marks_insufficient_hit_depth(tmp_path: Path) -> None:
    task = _make_task(tmp_path, evaluation=["hit_at_2"])
    query_representations = [
        RetrievalRepresentation(
            sample_id="query-1",
            representation_type="single_vector",
            data=[1.0],
            metadata=RetrievalItemMetadata(category="tumor", patient_id="patient-q1"),
        )
    ]
    results = [
        SearchResult(
            query_id="query-1",
            metadata=RetrievalItemMetadata(category="tumor", patient_id="patient-q1"),
            hits=[
                SearchHit(
                    item_id="ref-1",
                    score=0.99,
                    rank=1,
                    metadata=RetrievalItemMetadata(
                        category="tumor",
                        patient_id="patient-r1",
                    ),
                )
            ],
        )
    ]

    metrics = task._evaluate_results(
        results=results,
        query_representations=query_representations,
    )

    assert metrics["hit_at_2"]["micro"] == pytest.approx(0.0)
    assert metrics["hit_at_2"]["insufficient_k_queries"] == 1
