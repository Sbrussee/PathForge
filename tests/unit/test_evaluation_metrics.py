from __future__ import annotations

import pytest

from pathbench.core.evaluation.metrics.slide_retrieval import (
    compute_hit_at_k,
    compute_map_at_k,
    compute_mmv_at_k,
)
from pathbench.core.evaluation.tasks.slide_retrieval import (
    SlideRetrievalEvaluationData,
    SlideRetrievalEvaluationHit,
    SlideRetrievalEvaluationQuery,
)
from pathbench.core.evaluation.types import MetricRequest


def test_compute_hit_at_k_returns_macro_micro_and_counts() -> None:
    evaluation_data = SlideRetrievalEvaluationData(
        queries=[
            SlideRetrievalEvaluationQuery(
                query_id="q1",
                query_label="tumor",
                hits=[
                    SlideRetrievalEvaluationHit(
                        sample_id="h1",
                        label="tumor",
                        score=0.9,
                        rank=1,
                    )
                ],
            ),
            SlideRetrievalEvaluationQuery(
                query_id="q2",
                query_label="normal",
                hits=[
                    SlideRetrievalEvaluationHit(
                        sample_id="h2",
                        label="tumor",
                        score=0.8,
                        rank=1,
                    ),
                    SlideRetrievalEvaluationHit(
                        sample_id="h3",
                        label="normal",
                        score=0.7,
                        rank=2,
                    ),
                ],
            ),
        ]
    )

    payload = compute_hit_at_k(
        evaluation_data,
        request=MetricRequest(
            raw_name="hit_at_1",
            canonical_name="hit_at_k",
            params={"k": 1},
        ),
    )

    assert payload["macro"] == pytest.approx(0.5)
    assert payload["micro"] == pytest.approx(0.5)
    assert payload["per_label"] == {"normal": 0.0, "tumor": 1.0}
    assert payload["counts"]["num_queries"] == 2
    assert payload["counts"]["num_evaluable_queries"] == 2
    assert payload["counts_per_label"] == {"normal": 1, "tumor": 1}


def test_compute_hit_at_k_keeps_stable_payload_shape_for_larger_k() -> None:
    evaluation_data = SlideRetrievalEvaluationData(
        queries=[
            SlideRetrievalEvaluationQuery(
                query_id="q1",
                query_label="tumor",
                hits=[],
            )
        ]
    )

    payload = compute_hit_at_k(
        evaluation_data,
        request=MetricRequest(
            raw_name="hit_at_5",
            canonical_name="hit_at_k",
            params={"k": 5},
        ),
    )

    assert set(payload) == {
        "k",
        "macro",
        "micro",
        "per_label",
        "counts",
        "counts_per_label",
    }
    assert payload["k"] == 5


def test_compute_mmv_at_k_returns_expected_majority_vote_scores() -> None:
    evaluation_data = SlideRetrievalEvaluationData(
        queries=[
            SlideRetrievalEvaluationQuery(
                query_id="q1",
                query_label="tumor",
                hits=[
                    SlideRetrievalEvaluationHit(
                        sample_id="h1",
                        label="tumor",
                        score=0.9,
                        rank=1,
                    ),
                    SlideRetrievalEvaluationHit(
                        sample_id="h2",
                        label="tumor",
                        score=0.8,
                        rank=2,
                    ),
                    SlideRetrievalEvaluationHit(
                        sample_id="h3",
                        label="normal",
                        score=0.7,
                        rank=3,
                    ),
                ],
            ),
            SlideRetrievalEvaluationQuery(
                query_id="q2",
                query_label="normal",
                hits=[
                    SlideRetrievalEvaluationHit(
                        sample_id="h4",
                        label="tumor",
                        score=0.85,
                        rank=1,
                    ),
                    SlideRetrievalEvaluationHit(
                        sample_id="h5",
                        label="tumor",
                        score=0.75,
                        rank=2,
                    ),
                ],
            ),
        ]
    )

    payload = compute_mmv_at_k(
        evaluation_data,
        request=MetricRequest(
            raw_name="mmv_at_2",
            canonical_name="mmv_at_k",
            params={"k": 2},
        ),
    )

    assert payload["k"] == 2
    assert payload["per_label"] == {"normal": 0.0, "tumor": 1.0}
    assert payload["macro"] == pytest.approx(0.5)
    assert payload["micro"] == pytest.approx(0.5)
    assert payload["counts_per_label"] == {"normal": 1, "tumor": 1}


def test_compute_map_at_k_returns_expected_average_precision_scores() -> None:
    evaluation_data = SlideRetrievalEvaluationData(
        queries=[
            SlideRetrievalEvaluationQuery(
                query_id="q1",
                query_label="tumor",
                hits=[
                    SlideRetrievalEvaluationHit(
                        sample_id="h1",
                        label="tumor",
                        score=0.9,
                        rank=1,
                    ),
                    SlideRetrievalEvaluationHit(
                        sample_id="h2",
                        label="normal",
                        score=0.8,
                        rank=2,
                    ),
                    SlideRetrievalEvaluationHit(
                        sample_id="h3",
                        label="tumor",
                        score=0.7,
                        rank=3,
                    ),
                ],
            ),
            SlideRetrievalEvaluationQuery(
                query_id="q2",
                query_label="normal",
                hits=[
                    SlideRetrievalEvaluationHit(
                        sample_id="h4",
                        label="tumor",
                        score=0.85,
                        rank=1,
                    ),
                    SlideRetrievalEvaluationHit(
                        sample_id="h5",
                        label="normal",
                        score=0.75,
                        rank=2,
                    ),
                    SlideRetrievalEvaluationHit(
                        sample_id="h6",
                        label="normal",
                        score=0.65,
                        rank=3,
                    ),
                ],
            ),
        ]
    )

    payload = compute_map_at_k(
        evaluation_data,
        request=MetricRequest(
            raw_name="map_at_3",
            canonical_name="map_at_k",
            params={"k": 3},
        ),
    )

    assert payload["k"] == 3
    assert payload["per_label"]["tumor"] == pytest.approx((1.0 + (2.0 / 3.0)) / 2.0)
    assert payload["per_label"]["normal"] == pytest.approx(((1.0 / 2.0) + (2.0 / 3.0)) / 2.0)
    assert payload["macro"] == pytest.approx(
        (
            ((1.0 + (2.0 / 3.0)) / 2.0)
            + (((1.0 / 2.0) + (2.0 / 3.0)) / 2.0)
        )
        / 2.0
    )
    assert payload["counts_per_label"] == {"normal": 1, "tumor": 1}


def test_compute_map_at_k_counts_short_rankings_as_zero_ap() -> None:
    evaluation_data = SlideRetrievalEvaluationData(
        queries=[
            SlideRetrievalEvaluationQuery(
                query_id="q1",
                query_label="tumor",
                hits=[
                    SlideRetrievalEvaluationHit(
                        sample_id="h1",
                        label="tumor",
                        score=0.9,
                        rank=1,
                    )
                ],
            )
        ]
    )

    payload = compute_map_at_k(
        evaluation_data,
        request=MetricRequest(
            raw_name="map_at_2",
            canonical_name="map_at_k",
            params={"k": 2},
        ),
    )

    assert payload["per_label"] == {"tumor": 0.0}
    assert payload["micro"] == pytest.approx(0.0)
