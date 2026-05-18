from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pathbench.core.evaluation.slide_retrieval.metrics.f1 import (
    compute_macro_f1_at_k,
)
from pathbench.core.evaluation.slide_retrieval.metrics.hit import (
    compute_hit_at_k,
)
from pathbench.core.evaluation.slide_retrieval.metrics.map import (
    compute_map_at_k,
)
from pathbench.core.evaluation.slide_retrieval.metrics.mmv import (
    compute_mmv_at_k,
)
from pathbench.core.evaluation.slide_retrieval.metrics.ndcg import (
    compute_ndcg_at_k,
)
from pathbench.core.evaluation.slide_retrieval.metrics.precision import (
    compute_precision_at_k,
)
from pathbench.core.evaluation.slide_retrieval.data import (
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


def test_compute_mmv_at_k_requires_strict_majority_without_tie_break() -> None:
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
                ],
            )
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

    assert payload["per_label"] == {"tumor": 0.0}
    assert payload["macro"] == pytest.approx(0.0)
    assert payload["micro"] == pytest.approx(0.0)
    assert payload["counts"]["num_evaluable_queries"] == 1


def test_compute_hit_at_k_filters_query_self_hits() -> None:
    evaluation_data = SlideRetrievalEvaluationData(
        queries=[
            SlideRetrievalEvaluationQuery(
                query_id="q1",
                query_label="tumor",
                hits=[
                    SlideRetrievalEvaluationHit(
                        sample_id="q1",
                        label="tumor",
                        score=1.0,
                        rank=1,
                    ),
                    SlideRetrievalEvaluationHit(
                        sample_id="h1",
                        label="normal",
                        score=0.9,
                        rank=2,
                    ),
                ],
            )
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

    assert payload["per_label"] == {"tumor": 0.0}
    assert payload["macro"] == pytest.approx(0.0)


def test_compute_macro_f1_at_k_returns_expected_macro_f1() -> None:
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
                        score=0.95,
                        rank=1,
                    ),
                    SlideRetrievalEvaluationHit(
                        sample_id="h5",
                        label="normal",
                        score=0.85,
                        rank=2,
                    ),
                    SlideRetrievalEvaluationHit(
                        sample_id="h6",
                        label="tumor",
                        score=0.75,
                        rank=3,
                    ),
                ],
            ),
        ]
    )

    payload = compute_macro_f1_at_k(
        evaluation_data,
        request=MetricRequest(
            raw_name="macro_f1_at_3",
            canonical_name="macro_f1_at_k",
            params={"k": 3},
        ),
    )

    assert payload["k"] == 3
    assert payload["per_label"] == {"normal": 0.0, "tumor": 2.0 / 3.0}
    assert payload["macro"] == pytest.approx(1.0 / 3.0)
    assert payload["micro"] == pytest.approx(0.5)
    assert payload["counts"]["num_queries"] == 2
    assert payload["counts"]["num_evaluable_queries"] == 2
    assert payload["counts_per_label"] == {"normal": 1, "tumor": 1}


def test_compute_macro_f1_at_k_uses_available_hits_for_short_rankings() -> None:
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
                ],
            ),
            SlideRetrievalEvaluationQuery(
                query_id="q2",
                query_label="normal",
                hits=[
                    SlideRetrievalEvaluationHit(
                        sample_id="h3",
                        label="normal",
                        score=0.7,
                        rank=1,
                    ),
                    SlideRetrievalEvaluationHit(
                        sample_id="h4",
                        label="normal",
                        score=0.6,
                        rank=2,
                    ),
                ],
            ),
        ]
    )

    payload = compute_macro_f1_at_k(
        evaluation_data,
        request=MetricRequest(
            raw_name="macro_f1_at_2",
            canonical_name="macro_f1_at_k",
            params={"k": 2},
        ),
    )

    assert payload["per_label"] == {"normal": 1.0, "tumor": 0.0}
    assert payload["macro"] == pytest.approx(0.5)
    assert payload["micro"] == pytest.approx(2.0 / 3.0)
    assert payload["counts"]["num_evaluable_queries"] == 2


def test_compute_macro_f1_at_k_filters_query_self_hits_before_prediction() -> None:
    evaluation_data = SlideRetrievalEvaluationData(
        queries=[
            SlideRetrievalEvaluationQuery(
                query_id="q1",
                query_label="tumor",
                hits=[
                    SlideRetrievalEvaluationHit(
                        sample_id="q1",
                        label="tumor",
                        score=1.0,
                        rank=1,
                    ),
                    SlideRetrievalEvaluationHit(
                        sample_id="h1",
                        label="normal",
                        score=0.9,
                        rank=2,
                    ),
                ],
            )
        ]
    )

    payload = compute_macro_f1_at_k(
        evaluation_data,
        request=MetricRequest(
            raw_name="macro_f1_at_1",
            canonical_name="macro_f1_at_k",
            params={"k": 1},
        ),
    )

    assert payload["per_label"] == {"tumor": 0.0}
    assert payload["macro"] == pytest.approx(0.0)
    assert payload["counts"]["num_evaluable_queries"] == 1


def test_compute_precision_at_k_returns_expected_scores() -> None:
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
                        label="normal",
                        score=0.95,
                        rank=1,
                    ),
                    SlideRetrievalEvaluationHit(
                        sample_id="h5",
                        label="tumor",
                        score=0.85,
                        rank=2,
                    ),
                    SlideRetrievalEvaluationHit(
                        sample_id="h6",
                        label="normal",
                        score=0.75,
                        rank=3,
                    ),
                ],
            ),
        ]
    )

    payload = compute_precision_at_k(
        evaluation_data,
        request=MetricRequest(
            raw_name="precision_at_2",
            canonical_name="precision_at_k",
            params={"k": 2},
        ),
    )

    assert payload["k"] == 2
    assert payload["per_label"] == {"normal": 0.5, "tumor": 0.5}
    assert payload["macro"] == pytest.approx(0.5)
    assert payload["micro"] == pytest.approx(0.5)
    assert payload["counts"]["num_queries"] == 2
    assert payload["counts"]["num_evaluable_queries"] == 2
    assert payload["counts_per_label"] == {"normal": 1, "tumor": 1}


def test_compute_precision_at_k_penalizes_short_rankings_by_requested_k() -> None:
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
                ],
            )
        ]
    )

    payload = compute_precision_at_k(
        evaluation_data,
        request=MetricRequest(
            raw_name="precision_at_3",
            canonical_name="precision_at_k",
            params={"k": 3},
        ),
    )

    assert payload["per_label"] == {"tumor": 1.0 / 3.0}
    assert payload["macro"] == pytest.approx(1.0 / 3.0)
    assert payload["micro"] == pytest.approx(1.0 / 3.0)


def test_compute_precision_at_k_filters_query_self_hits() -> None:
    evaluation_data = SlideRetrievalEvaluationData(
        queries=[
            SlideRetrievalEvaluationQuery(
                query_id="q1",
                query_label="tumor",
                hits=[
                    SlideRetrievalEvaluationHit(
                        sample_id="q1",
                        label="tumor",
                        score=1.0,
                        rank=1,
                    ),
                    SlideRetrievalEvaluationHit(
                        sample_id="h1",
                        label="tumor",
                        score=0.9,
                        rank=2,
                    ),
                    SlideRetrievalEvaluationHit(
                        sample_id="h2",
                        label="normal",
                        score=0.8,
                        rank=3,
                    ),
                ],
            )
        ]
    )

    payload = compute_precision_at_k(
        evaluation_data,
        request=MetricRequest(
            raw_name="precision_at_2",
            canonical_name="precision_at_k",
            params={"k": 2},
        ),
    )

    assert payload["per_label"] == {"tumor": 0.5}
    assert payload["macro"] == pytest.approx(0.5)


def test_compute_ndcg_at_k_uses_findable_reference_count_for_idcg(
    tmp_path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "annotations.csv").write_text(
        "dataset,slide,case,patient,category\n"
        "query_ds,S1,C1,P1,tumor\n"
        "ref_ds,S2,C2,P2,tumor\n"
        "ref_ds,S3,C3,P3,tumor\n"
        "ref_ds,S4,C4,P4,normal\n"
        "query_ds,S5,C5,P5,tumor\n",
        encoding="utf-8",
    )
    run_dir = project_root / "eval_slide_retrieval" / "combo" / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)

    evaluation_data = SlideRetrievalEvaluationData(
        queries=[
            SlideRetrievalEvaluationQuery(
                query_id="S1",
                query_label="tumor",
                hits=[
                    SlideRetrievalEvaluationHit(
                        sample_id="S2",
                        label="tumor",
                        score=0.9,
                        rank=1,
                    ),
                    SlideRetrievalEvaluationHit(
                        sample_id="S4",
                        label="normal",
                        score=0.8,
                        rank=2,
                    ),
                    SlideRetrievalEvaluationHit(
                        sample_id="S3",
                        label="tumor",
                        score=0.7,
                        rank=3,
                    ),
                ],
            )
        ]
    )
    run_context = SimpleNamespace(
        run_dir=run_dir,
        aggregation_level="slide",
        label_column="category",
        manifest={
            "exclusion_level": "patient",
            "reference_dataset_names": ["ref_ds"],
        },
    )

    payload = compute_ndcg_at_k(
        evaluation_data,
        request=MetricRequest(
            raw_name="ndcg_at_3",
            canonical_name="ndcg_at_k",
            params={"k": 3},
        ),
        run_context=run_context,
    )

    expected_dcg = 1.0 + (1.0 / 2.0)
    expected_idcg = 1.0 + (1.0 / np.log2(3.0))
    assert payload["k"] == 3
    assert payload["per_label"] == {
        "tumor": pytest.approx(expected_dcg / expected_idcg)
    }
    assert payload["macro"] == pytest.approx(expected_dcg / expected_idcg)
    assert payload["micro"] == pytest.approx(expected_dcg / expected_idcg)
    assert payload["counts"]["num_queries"] == 1
    assert payload["counts"]["num_evaluable_queries"] == 1


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
