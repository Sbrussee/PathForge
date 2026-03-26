from __future__ import annotations

import numpy as np

from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathbench.slide_retrieval.search_strategies.strategies.yottixel import (
    BoB,
    YottixelSearch,
)
from pathbench.slide_retrieval.types import RetrievalItemMetadata


def _make_representation(
    *,
    sample_id: str,
    data: np.ndarray,
    category: str,
    patient_id: str,
    representation_type: str = "multi_vector",
) -> RetrievalRepresentation:
    """
    Build one retrieval representation for Yottixel search tests.

    Inputs:
        sample_id:
            Retrieval item identifier.
        data:
            Retrieval array with shape ``(N, D)`` or ``(D,)``.
        category:
            Ground-truth category label.
        patient_id:
            Patient identifier used for filtering.
        representation_type:
            Retrieval representation kind for search validation.

    Output:
        Returns a ``RetrievalRepresentation`` with normalized metadata.
    """
    return RetrievalRepresentation(
        sample_id=sample_id,
        representation_type=representation_type,
        data=np.asarray(data, dtype=np.float32),
        metadata=RetrievalItemMetadata(
            category=category,
            patient_id=patient_id,
        ),
    )


def test_bob_distance_uses_median_of_minimum_xor_distances() -> None:
    left = BoB(
        barcodes=np.array([[1, 1], [1, 0]], dtype=np.uint8),
        slide_id="left",
        patient_id="patient-left",
        label="A",
    )
    right = BoB(
        barcodes=np.array([[1, 1], [0, 0]], dtype=np.uint8),
        slide_id="right",
        patient_id="patient-right",
        label="B",
    )

    distance = left.distance(right)

    assert distance == 0.5


def test_yottixel_search_ranks_reference_items_by_bob_distance() -> None:
    reference_a = _make_representation(
        sample_id="ref-a",
        data=np.array([[3.0, 2.0, 1.0], [4.0, 3.0, 2.0]], dtype=np.float32),
        category="tumor",
        patient_id="patient-a",
    )
    reference_b = _make_representation(
        sample_id="ref-b",
        data=np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]], dtype=np.float32),
        category="normal",
        patient_id="patient-b",
    )
    query = _make_representation(
        sample_id="query",
        data=np.array([[5.0, 4.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32),
        category="tumor",
        patient_id="patient-q",
    )

    strategy = YottixelSearch(params={"k": 2})
    strategy.build_database([reference_a, reference_b])
    result = strategy.search(query)

    assert result.query_id == "query"
    assert result.metadata["predicted_category"] == "tumor"
    assert [hit.item_id for hit in result.hits] == ["ref-a", "ref-b"]
    assert [hit.rank for hit in result.hits] == [1, 2]
    assert [hit.metadata.category for hit in result.hits] == ["tumor", "normal"]
    assert result.hits[0].score == 0.0
    assert result.hits[1].score == 2.0


def test_yottixel_search_filters_same_patient_and_falls_back_when_empty() -> None:
    reference = _make_representation(
        sample_id="ref-a",
        data=np.array([[3.0, 2.0, 1.0]], dtype=np.float32),
        category="tumor",
        patient_id="patient-1",
    )
    query = _make_representation(
        sample_id="query",
        data=np.array([[5.0, 4.0, 3.0]], dtype=np.float32),
        category="tumor",
        patient_id="patient-1",
    )

    strategy = YottixelSearch(params={"k": 3})
    strategy.build_database([reference])
    result = strategy.search(query, filter_same_patient=True)

    assert result.hits == []
    assert result.metadata["predicted_category"] == "tumor"
    assert result.metadata["top_k_labels"] == []


def test_yottixel_search_supports_single_vector_inputs() -> None:
    reference = _make_representation(
        sample_id="ref-a",
        data=np.array([3.0, 2.0, 1.0], dtype=np.float32),
        category="tumor",
        patient_id="patient-a",
        representation_type="single_vector",
    )
    query = _make_representation(
        sample_id="query",
        data=np.array([4.0, 3.0, 2.0], dtype=np.float32),
        category="tumor",
        patient_id="patient-q",
        representation_type="single_vector",
    )

    strategy = YottixelSearch(params={"k": 1})
    strategy.build_database([reference])
    result = strategy.search(query)

    assert [hit.item_id for hit in result.hits] == ["ref-a"]
    assert result.hits[0].score == 0.0
