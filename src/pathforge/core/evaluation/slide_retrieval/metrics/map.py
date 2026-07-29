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
from pathforge.core.evaluation.slide_retrieval.pool import (
    count_relevant_reference_items_by_query,
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

    k = int(request.params["k"])
    if k <= 0:
        raise ValueError(f"Expected k > 0 for map_at_k. Got {k}.")

    relevant_counts = count_relevant_reference_items_by_query(
        evaluation_data=evaluation_data,
        run_context=run_context,
    )
    ap_per_label: dict[str, list[float]] = defaultdict(list)
    counts_per_label: dict[str, int] = defaultdict(int)

    for query in evaluation_data.queries:
        relevant_available_count = (
            None if relevant_counts is None else relevant_counts[str(query.query_id)]
        )
        if relevant_available_count is not None and relevant_available_count <= 0:
            continue
        counts_per_label[query.query_label] += 1
        top_hits = get_top_k_hits(query, k=k)
        if relevant_available_count is None and len(top_hits) < k:
            ap_per_label[query.query_label].append(0.0)
            continue

        num_relevant = 0
        precision_sum = 0.0
        for index, hit in enumerate(top_hits, start=1):
            if hit.label == query.query_label:
                num_relevant += 1
                precision_sum += float(num_relevant) / float(index)

        denominator = (
            num_relevant
            if relevant_available_count is None
            else min(k, relevant_available_count)
        )
        ap_per_label[query.query_label].append(
            (precision_sum / float(denominator)) if denominator > 0 else 0.0
        )

    per_label = {
        label: (float(np.mean(ap_values)) if ap_values else 0.0)
        for label, ap_values in sorted(ap_per_label.items())
    }
    return {
        "k": k,
        **build_label_aggregate_payload(
            per_label_values=per_label,
            counts_per_label=counts_per_label,
            evaluable_queries=sum(counts_per_label.values()),
            total_queries=len(evaluation_data.queries),
        ),
    }
