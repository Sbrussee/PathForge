from __future__ import annotations

import numpy as np
import pytest

from pathforge.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathforge.slide_retrieval.search_strategies.registry import (
    get_search_strategy_hyperparams,
)
from pathforge.slide_retrieval.search_strategies.strategies.retccl import (
    RetCCLSearch,
    _safe_mean,
)
from pathforge.slide_retrieval.types import RetrievalItemMetadata


def _make_representation(
    *,
    sample_id: str,
    patient_id: str,
    category: str | None,
    features: list[list[float]] | np.ndarray,
) -> RetrievalRepresentation:
    """
    Build one minimal multi-vector retrieval representation for RETCCL tests.

    Inputs:
        sample_id (str): Retrieval item identifier.
        patient_id (str): Patient identifier.
        category (str | None): Slide label, or missing metadata.
        features (list[list[float]] | np.ndarray): Matrix with shape ``(N, D)``.

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
            "topk_per_patch": 2,
        }
    )
    strategy.build_database(reference_representations)

    result = strategy.search(query_representation=query_representation)

    assert result.query_id == "query-slide"
    assert [hit.item_id for hit in result.hits] == ["slide-a", "slide-b"]
    assert [hit.metadata.category for hit in result.hits] == ["class-a", "class-b"]
    assert [hit.rank for hit in result.hits] == [1, 2]


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


@pytest.mark.parametrize("k, expected_ids", [(1, ["mixed"]), (5, ["mixed", "pure"])])
def test_retccl_preserves_entropy_order_over_similarity(
    k: int, expected_ids: list[str]
) -> None:
    """A retained mixed-label bag precedes a more similar single-label bag."""
    strategy = RetCCLSearch(params={"k": k})
    strategy.build_database(
        [
            _make_representation(
                sample_id="mixed",
                patient_id="p1",
                category="a",
                features=[[0.85, 0.0, np.sqrt(1 - 0.85**2)]],
            ),
            _make_representation(
                sample_id="other",
                patient_id="p2",
                category="b",
                features=[[0.8, 0.0, 0.6]],
            ),
            _make_representation(
                sample_id="pure",
                patient_id="p3",
                category="c",
                features=[[0.0, 1.0, 0.0]],
            ),
        ]
    )
    # The unmatched third patch lowers eta, so both nonempty bags survive.
    query = _make_representation(
        sample_id="query",
        patient_id="pq",
        category="query",
        features=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
    )

    result = strategy.search(query)

    assert [hit.item_id for hit in result.hits] == expected_ids
    assert [hit.rank for hit in result.hits] == list(range(1, len(expected_ids) + 1))
    assert [hit.score for hit in result.hits] == pytest.approx(
        [0.825, 1.0][: len(expected_ids)]
    )


def test_retccl_frequency_weights_determine_bag_order() -> None:
    """Equation 9 favors a balanced weighted bag under database frequencies."""
    references = [
        _make_representation(
            sample_id="x-a",
            patient_id="pxa",
            category="a",
            features=[[1.0, 0.0, 0.0]],
        ),
        _make_representation(
            sample_id="x-b",
            patient_id="pxb",
            category="b",
            features=[[1.0, 0.0, 0.0]],
        ),
        _make_representation(
            sample_id="y-a",
            patient_id="pya",
            category="a",
            features=[[0.0, 1.0, 0.0]],
        ),
        _make_representation(
            sample_id="y-b",
            patient_id="pyb",
            category="b",
            features=[[0.0, 1.0, 0.0]] * 4,
        ),
    ]
    references.extend(
        _make_representation(
            sample_id=f"unused-{i}",
            patient_id=f"pu-{i}",
            category="a",
            features=[[0.0, 0.0, 1.0]],
        )
        for i in range(6)
    )
    strategy = RetCCLSearch()
    strategy.build_database(references)
    query = _make_representation(
        sample_id="query",
        patient_id="pq",
        category="query",
        features=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    result = strategy.search(query)

    # Frequencies are 0.8/0.2 by slide, not by patch. The y bag's 1:4
    # patch ratio becomes 1:1 after weighting and has maximal entropy.
    assert strategy.class_weight == pytest.approx({"a": 0.8, "b": 0.2})
    assert [hit.item_id for hit in result.hits] == ["y-b", "x-a"]


@pytest.mark.parametrize(
    "categories, expected",
    [
        (["a", "a", "b"], {"a": 2 / 3, "b": 1 / 3}),
        (["a"], {"a": 1.0}),
        ([], {}),
        ([None, None], {None: 1.0}),
        ([None, "a"], {None: 0.5, "a": 0.5}),
    ],
)
def test_retccl_class_weights_are_normalized_database_frequencies(
    categories: list[str | None],
    expected: dict[str | None, float],
) -> None:
    """Diagnosis frequencies are determined directly from reference items."""
    strategy = RetCCLSearch()
    strategy.build_database(
        [
            _make_representation(
                sample_id=f"s{i}",
                patient_id=f"p{i}",
                category=category,
                features=[[1.0, 0.0]],
            )
            for i, category in enumerate(categories)
        ]
    )

    assert strategy.class_weight == pytest.approx(expected)


def test_retccl_does_not_expose_class_weight_factor() -> None:
    """Configuration and optimization schemas must omit the removed parameter."""
    strategy = RetCCLSearch()
    assert "class_weight_factor" not in get_search_strategy_hyperparams("retccl")
    assert "class_weight_factor" not in strategy.hyperparam_values()
    assert not hasattr(strategy, "class_weight_factor")


def test_retccl_keeps_first_nomination_score_for_duplicate_slide() -> None:
    """A later pure bag does not replace the first mixed bag's score."""
    strategy = RetCCLSearch()
    strategy.build_database(
        [
            _make_representation(
                sample_id="repeated",
                patient_id="p1",
                category="a",
                features=[[0.85, 0.0, np.sqrt(1 - 0.85**2)], [0.0, 1.0, 0.0]],
            ),
            _make_representation(
                sample_id="other",
                patient_id="p2",
                category="b",
                features=[[0.8, 0.0, 0.6]],
            ),
        ]
    )
    query = _make_representation(
        sample_id="query",
        patient_id="pq",
        category="query",
        features=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
    )

    result = strategy.search(query)

    assert [hit.item_id for hit in result.hits] == ["repeated"]
    assert result.hits[0].score == pytest.approx(0.825)


@pytest.mark.parametrize("category", ["a", None])
def test_retccl_handles_missing_categories_without_invalid_entropy(
    category: str | None,
) -> None:
    """Missing labels form one unknown category for compatibility."""
    strategy = RetCCLSearch()
    strategy.build_database(
        [
            _make_representation(
                sample_id="s1",
                patient_id="p1",
                category=category,
                features=[[1.0, 0.0]],
            ),
            _make_representation(
                sample_id="s2",
                patient_id="p2",
                category=None,
                features=[[0.95, 0.05]],
            ),
        ]
    )
    query = _make_representation(
        sample_id="query",
        patient_id="pq",
        category="query",
        features=[[1.0, 0.0]],
    )

    with np.errstate(divide="raise", invalid="raise"):
        result = strategy.search(query)

    assert [hit.item_id for hit in result.hits] == ["s1"]
    assert np.isfinite(result.hits[0].score)


@pytest.mark.parametrize(
    "features", [np.empty((0, 2)), np.zeros((1, 2)), [[0.0, 1.0]]]
)
def test_retccl_returns_no_hits_for_empty_zero_or_unmatched_query(
    features: list[list[float]] | np.ndarray,
) -> None:
    """Queries without accepted matches yield no votes or hits."""
    strategy = RetCCLSearch()
    strategy.build_database(
        [
            _make_representation(
                sample_id="s1",
                patient_id="p1",
                category="a",
                features=[[1.0, 0.0]],
            )
        ]
    )
    query = _make_representation(
        sample_id="query",
        patient_id="pq",
        category="query",
        features=features,
    )

    assert strategy.search(query).hits == []


@pytest.mark.parametrize("values, expected", [([0.5, 1.0], 0.75), ([], 0.0)])
def test_retccl_safe_mean(values: list[float], expected: float) -> None:
    """Empty bags contribute zero to the eta threshold."""
    assert _safe_mean(values) == expected


@pytest.mark.parametrize("features", [[1.0, 2.0], [[1.0, 2.0]]])
def test_retccl_accepts_one_vector_or_matrix(
    features: list[float] | list[list[float]],
) -> None:
    """Both supported shapes normalize to one row without changing values."""
    result = RetCCLSearch()._as_feature_matrix(
        representation_data=features, item_id="q"
    )
    np.testing.assert_array_equal(result, [[1.0, 2.0]])


@pytest.mark.parametrize("features", [1.0, [[[1.0, 2.0]]]])
def test_retccl_rejects_invalid_feature_dimensions(features: object) -> None:
    """Scalars and tensors beyond two dimensions cannot represent patch bags."""
    with pytest.raises(ValueError, match="expects retrieval items with shape"):
        RetCCLSearch()._as_feature_matrix(representation_data=features, item_id="q")
