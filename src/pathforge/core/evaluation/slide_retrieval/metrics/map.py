from __future__ import annotations

from collections import defaultdict

import numpy as np

from pathforge.core.evaluation.registry import evaluation_metric
from pathforge.core.evaluation.slide_retrieval.data import (
    SlideRetrievalEvaluationData,
)
from pathforge.core.evaluation.slide_retrieval.metrics import (
    build_label_aggregate_payload,
)
from pathforge.core.evaluation.slide_retrieval.voting import get_top_k_hits
from pathforge.core.evaluation.types import MetricRequest


@evaluation_metric(
    "map_at_k",
    tasks=("slide_retrieval",),
    pattern=r"^map_at_(?P<k>[1-9]\d*)$",
    param_builder=lambda match: {"k": int(match.group("k"))},
)
def compute_map_at_k(
    evaluation_data: SlideRetrievalEvaluationData,
    *,
    request: MetricRequest,
    run_context: object | None = None,
) -> dict[str, object]:
    """Compute mean average precision at the requested `k`."""

    _ = run_context

    k = int(request.params["k"])
    if k <= 0:
        raise ValueError(f"Expected k > 0 for map_at_k. Got {k}.")

    ap_per_label: dict[str, list[float]] = defaultdict(list)
    counts_per_label: dict[str, int] = defaultdict(int)

    for query in evaluation_data.queries:
        counts_per_label[query.query_label] += 1
        top_hits = get_top_k_hits(query, k=k)
        if len(top_hits) < k:
            ap_per_label[query.query_label].append(0.0)
            continue

        num_relevant = 0
        precisions: list[float] = []
        for index, hit in enumerate(top_hits, start=1):
            if hit.label == query.query_label:
                num_relevant += 1
                precisions.append(float(num_relevant) / float(index))

        ap_per_label[query.query_label].append(
            float(np.mean(precisions)) if precisions else 0.0
        )

    per_label = {
        label: (
            float(np.mean(ap_values))
            if ap_values
            else 0.0
        )
        for label, ap_values in sorted(ap_per_label.items())
    }
    return {
        "k": k,
        **build_label_aggregate_payload(
            per_label_values=per_label,
            counts_per_label=counts_per_label,
            evaluable_queries=sum(counts_per_label.values()),
        ),
    }
