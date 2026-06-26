from __future__ import annotations

from collections import defaultdict

from pathforge.core.evaluation.registry import evaluation_metric
from pathforge.core.evaluation.slide_retrieval.data import (
    SlideRetrievalEvaluationData,
)
from pathforge.core.evaluation.slide_retrieval.metrics import (
    build_label_aggregate_payload,
)
from pathforge.core.evaluation.slide_retrieval.voting import predict_label_from_top_k
from pathforge.core.evaluation.types import MetricRequest


@evaluation_metric(
    "mmv_at_k",
    tasks=("slide_retrieval",),
    pattern=r"^mmv_at_(?P<k>[1-9]\d*)$",
    param_builder=lambda match: {"k": int(match.group("k"))},
)
def compute_mmv_at_k(
    evaluation_data: SlideRetrievalEvaluationData,
    *,
    request: MetricRequest,
    run_context: object | None = None,
) -> dict[str, object]:
    """Compute majority-vote accuracy at the requested `k`."""

    _ = run_context

    k = int(request.params["k"])
    if k <= 0:
        raise ValueError(f"Expected k > 0 for mmv_at_k. Got {k}.")

    correct_counts: dict[str, int] = defaultdict(int)
    counts_per_label: dict[str, int] = defaultdict(int)

    for query in evaluation_data.queries:
        counts_per_label[query.query_label] += 1
        predicted_label = predict_label_from_top_k(query, k=k)
        if predicted_label == query.query_label:
            correct_counts[query.query_label] += 1

    per_label = {
        label: (
            float(correct_counts[label]) / float(count)
            if count > 0
            else 0.0
        )
        for label, count in sorted(counts_per_label.items())
    }
    return {
        "k": k,
        **build_label_aggregate_payload(
            per_label_values=per_label,
            counts_per_label=counts_per_label,
            evaluable_queries=sum(counts_per_label.values()),
        ),
    }
