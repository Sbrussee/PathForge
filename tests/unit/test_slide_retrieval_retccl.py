from __future__ import annotations

import numpy as np

from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathbench.slide_retrieval.search_strategies.strategies.retccl import RetCCLSearch
from pathbench.slide_retrieval.types import RetrievalItemMetadata


def _make_representation(
    *,
    sample_id: str,
    patient_id: str,
    category: str,
    features: list[list[float]],
) -> RetrievalRepresentation:
    """
    Build one minimal multi-vector retrieval representation for RETCCL tests.

    Inputs:
        sample_id (str): Retrieval item identifier.
        patient_id (str): Patient identifier.
        category (str): Slide label.
        features (list[list[float]]): Feature matrix with shape ``(N, D)``.

    Outputs:
        RetrievalRepresentation: Multi-vector representation with shape ``(N, D)``.

    Example:
        >>> representation = _make_representation(
        ...     sample_id="s1",
        ...     patient_id="p1",
        ...     category="a",
        ...     features=[[1.0, 0.0]],
        ... )
        >>> representation.data.shape
        (1, 2)
    """
    return RetrievalRepresentation(
        sample_id=sample_id,
        representation_type="multi_vector",
        data=np.asarray(features, dtype=np.float32),
        metadata=RetrievalItemMetadata(
            category=category,
            patient_id=patient_id,
            member_ids=[sample_id],
        ),
    )


def test_retccl_search_ranks_expected_hits_and_excludes_same_patient() -> None:
    reference_representations = [
        _make_representation(
            sample_id="same-patient",
            patient_id="patient-query",
            category="ignore",
            features=[[1.0, 0.0], [0.0, 1.0]],
        ),
        _make_representation(
            sample_id="slide-a",
            patient_id="patient-a",
            category="class-a",
            features=[[1.0, 0.0], [0.95, 0.05]],
        ),
        _make_representation(
            sample_id="slide-b",
            patient_id="patient-b",
            category="class-b",
            features=[[0.0, 1.0], [0.05, 0.95]],
        ),
    ]
    query_representation = _make_representation(
        sample_id="query-slide",
        patient_id="patient-query",
        category="query-class",
        features=[[1.0, 0.0], [0.0, 1.0]],
    )

    strategy = RetCCLSearch(
        params={
            "k": 2,
            "cosine_threshold": 0.7,
            "class_weight_factor": 10.0,
            "topk_per_patch": 2,
        }
    )
    strategy.build_database(reference_representations)

    result = strategy.search(query_representation=query_representation)

    assert result.query_id == "query-slide"
    assert [hit.item_id for hit in result.hits] == ["slide-a", "slide-b"]
    assert [hit.metadata.category for hit in result.hits] == ["class-a", "class-b"]
    assert [hit.rank for hit in result.hits] == [1, 2]
    assert result.hits[0].score > result.hits[1].score


def test_retccl_search_returns_no_hits_when_no_candidate_patches_remain() -> None:
    reference_representations = [
        _make_representation(
            sample_id="same-patient",
            patient_id="patient-query",
            category="class-a",
            features=[[1.0, 0.0]],
        )
    ]
    query_representation = _make_representation(
        sample_id="query-slide",
        patient_id="patient-query",
        category="query-class",
        features=[[1.0, 0.0]],
    )

    strategy = RetCCLSearch(params={"k": 3})
    strategy.build_database(reference_representations)

    result = strategy.search(query_representation=query_representation)

    assert result.query_id == "query-slide"
    assert result.hits == []
