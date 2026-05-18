from __future__ import annotations

from collections import defaultdict

from pathbench.core.evaluation.registry import evaluation_metric
from pathbench.core.evaluation.slide_retrieval.data import (
    SlideRetrievalEvaluationData,
)
from pathbench.core.evaluation.slide_retrieval.metrics import (
    build_label_aggregate_payload,
)
from pathbench.core.evaluation.slide_retrieval.voting import get_top_k_hits
from pathbench.core.evaluation.types import MetricRequest


@evaluation_metric(
    "hit_at_k",
    tasks=("slide_retrieval",),
    pattern=r"^hit_at_(?P<k>[1-9]\d*)$",
    param_builder=lambda match: {"k": int(match.group("k"))},
)
def compute_hit_at_k(
    evaluation_data: SlideRetrievalEvaluationData,
    *,
    request: MetricRequest,
    run_context: object | None = None,
) -> dict[str, object]:
    """Compute slide-retrieval hit-rate at the requested `k`."""

    _ = run_context

    k = int(request.params["k"])
    if k <= 0:
        raise ValueError(f"Expected k > 0 for hit_at_k. Got {k}.")

    per_label_hits: dict[str, int] = defaultdict(int)
    counts_per_label: dict[str, int] = defaultdict(int)

    for query in evaluation_data.queries:
        counts_per_label[query.query_label] += 1
        top_hits = get_top_k_hits(query, k=k)
        if any(hit.label == query.query_label for hit in top_hits):
            per_label_hits[query.query_label] += 1

    per_label = {
        label: (
            float(per_label_hits[label]) / float(count)
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
