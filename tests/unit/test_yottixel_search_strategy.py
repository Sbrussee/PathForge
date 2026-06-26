from __future__ import annotations

import numpy as np

from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathbench.slide_retrieval.search_strategies.strategies.yottixel import (
    BoB,
    YottixelSearch,
)


def _make_representation(
    *,
    sample_id: str,
    data: np.ndarray,
    exclusion_key: str | None = None,
) -> RetrievalRepresentation:
    """
    Build one retrieval representation for Yottixel search tests.

    Inputs:
    - `sample_id`: retrieval item identifier.
    - `data`: retrieval array with shape `(N, D)` or `(D,)`.
    - `exclusion_key`: optional exclusion key used to filter reference items.

    Returns:
    - `RetrievalRepresentation` with the benchmark-time runtime contract only.
    """
    return RetrievalRepresentation(
        sample_id=sample_id,
        data=np.asarray(data, dtype=np.float32),
        exclusion_key=exclusion_key,
    )


def test_bob_distance_uses_median_of_minimum_xor_distances() -> None:
    left = BoB(
        barcodes=np.array([[1, 1], [1, 0]], dtype=np.uint8),
        slide_id="left",
        exclusion_key="patient-left",
    )
    right = BoB(
        barcodes=np.array([[1, 1], [0, 0]], dtype=np.uint8),
        slide_id="right",
        exclusion_key="patient-right",
    )

    distance = left.distance(right)

    assert distance == 0.5


def test_bob_distance_matches_brute_force_xor_distance() -> None:
    rng = np.random.default_rng(7)
    left = BoB(
        barcodes=rng.integers(0, 2, size=(13, 31), dtype=np.uint8),
        slide_id="left",
    )
    right = BoB(
        barcodes=rng.integers(0, 2, size=(17, 31), dtype=np.uint8),
        slide_id="right",
    )

    brute_force_distances = []
    for barcode in left.barcodes:
        distances = [
            int(np.count_nonzero(np.not_equal(barcode, other)))
            for other in right.barcodes
        ]
        brute_force_distances.append(min(distances))

    assert left.distance(right) == float(np.median(brute_force_distances))


def test_yottixel_search_ranks_reference_items_by_bob_distance() -> None:
    reference_a = _make_representation(
        sample_id="ref-a",
        data=np.array([[3.0, 2.0, 1.0], [4.0, 3.0, 2.0]], dtype=np.float32),
    )
    reference_b = _make_representation(
        sample_id="ref-b",
        data=np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 2.0]], dtype=np.float32),
    )
    query = _make_representation(
        sample_id="query",
        data=np.array([[5.0, 4.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32),
    )

    strategy = YottixelSearch(params={"k": 2})
    strategy.build_database([reference_a, reference_b])
    result = strategy.search(query)

    assert result.query_id == "query"
    assert [hit.item_id for hit in result.hits] == ["ref-a", "ref-b"]
    assert [hit.rank for hit in result.hits] == [1, 2]
    assert result.hits[0].score == 0.0
    assert result.hits[1].score == 2.0


def test_yottixel_reuses_database_item_for_shared_query_reference_sample() -> None:
    representation = _make_representation(
        sample_id="shared",
        data=np.array([[3.0, 2.0, 1.0]], dtype=np.float32),
        exclusion_key="patient-1",
    )

    strategy = YottixelSearch(params={"k": 1})
    strategy.build_database([representation])

    query_item = strategy.prepare_query(representation)

    assert query_item is strategy.search_database[0]


def test_yottixel_prepares_all_queries_once_and_searches_prepared_items() -> None:
    shared = _make_representation(
        sample_id="shared",
        data=np.array([[3.0, 2.0, 1.0]], dtype=np.float32),
        exclusion_key="patient-1",
    )
    reference = _make_representation(
        sample_id="ref",
        data=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        exclusion_key="patient-2",
    )
    query_only = _make_representation(
        sample_id="query-only",
        data=np.array([[4.0, 3.0, 2.0]], dtype=np.float32),
        exclusion_key="patient-3",
    )

    strategy = YottixelSearch(params={"k": 2})
    strategy.build_database([shared, reference])

    prepared_queries = strategy.prepare_queries([shared, query_only])
    results = [
        strategy.search_prepared(query_item=query_item)
        for query_item in prepared_queries
    ]

    assert prepared_queries[0] is strategy.search_database[0]
    assert prepared_queries[1].item_id == "query-only"
    assert [result.query_id for result in results] == ["shared", "query-only"]


def test_yottixel_search_filters_same_exclusion_key() -> None:
    reference = _make_representation(
        sample_id="ref-a",
        data=np.array([[3.0, 2.0, 1.0]], dtype=np.float32),
        exclusion_key="patient-1",
    )
    query = _make_representation(
        sample_id="query",
        data=np.array([[5.0, 4.0, 3.0]], dtype=np.float32),
        exclusion_key="patient-1",
    )

    strategy = YottixelSearch(params={"k": 3})
    strategy.build_database([reference])
    result = strategy.search(query)

    assert result.hits == []


def test_yottixel_search_supports_single_vector_inputs() -> None:
    reference = _make_representation(
        sample_id="ref-a",
        data=np.array([3.0, 2.0, 1.0], dtype=np.float32),
    )
    query = _make_representation(
        sample_id="query",
        data=np.array([4.0, 3.0, 2.0], dtype=np.float32),
    )

    strategy = YottixelSearch(params={"k": 1})
    strategy.build_database([reference])
    result = strategy.search(query)

    assert [hit.item_id for hit in result.hits] == ["ref-a"]
    assert result.hits[0].score == 0.0
