from __future__ import annotations

from collections import defaultdict

import numpy as np

from pathbench.core.evaluation.registry import evaluation_metric
from pathbench.core.evaluation.slide_retrieval.data import (
    SlideRetrievalEvaluationData,
)
from pathbench.core.evaluation.slide_retrieval.metrics import (
    build_label_aggregate_payload,
)
from pathbench.core.evaluation.slide_retrieval.pool import (
    build_aggregated_reference_pool,
    compute_dcg,
    count_relevant_reference_items_for_query,
)
from pathbench.core.evaluation.slide_retrieval.voting import get_top_k_hits
from pathbench.core.evaluation.types import MetricRequest


@evaluation_metric(
    "ndcg_at_k",
    tasks=("slide_retrieval",),
    pattern=r"^ndcg_at_(?P<k>[1-9]\d*)$",
    param_builder=lambda match: {"k": int(match.group("k"))},
)
def compute_ndcg_at_k(
    evaluation_data: SlideRetrievalEvaluationData,
    *,
    request: MetricRequest,
    run_context: object | None = None,
) -> dict[str, object]:
    """Compute binary NDCG at the requested `k`."""

    if run_context is None:
        raise ValueError("ndcg_at_k requires run_context to reconstruct the reference pool.")

    k = int(request.params["k"])
    if k <= 0:
        raise ValueError(f"Expected k > 0 for ndcg_at_k. Got {k}.")

    all_items_df, reference_pool_df = build_aggregated_reference_pool(
        run_context=run_context
    )
    ndcg_per_label: dict[str, list[float]] = defaultdict(list)
    counts_per_label: dict[str, int] = defaultdict(int)

    for query in evaluation_data.queries:
        counts_per_label[query.query_label] += 1

        top_hits = get_top_k_hits(query, k=k)
        observed_relevance = [
            1 if hit.label == query.query_label else 0
            for hit in top_hits
        ]
        dcg = compute_dcg(observed_relevance)

        relevant_available_count = count_relevant_reference_items_for_query(
            query=query,
            all_items_df=all_items_df,
            reference_pool_df=reference_pool_df,
        )
        ideal_relevant_count = min(k, relevant_available_count)
        ideal_relevance = [1] * ideal_relevant_count
        idcg = compute_dcg(ideal_relevance)

        ndcg_per_label[query.query_label].append(
            (dcg / idcg) if idcg > 0.0 else 0.0
        )

    per_label = {
        label: (
            float(np.mean(ndcg_values))
            if ndcg_values
            else 0.0
        )
        for label, ndcg_values in sorted(ndcg_per_label.items())
    }
    return {
        "k": k,
        **build_label_aggregate_payload(
            per_label_values=per_label,
            counts_per_label=counts_per_label,
            evaluable_queries=sum(counts_per_label.values()),
        ),
    }
