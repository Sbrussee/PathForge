from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from pathbench.slide_retrieval.validation.registry import register_validation_metric
from pathbench.slide_retrieval.validation.types import NormalizedSearchResult


@register_validation_metric("hit_at_k")
def compute_hit_at_k(
    results: Iterable[NormalizedSearchResult],
    *,
    k: int,
) -> dict[str, object]:
    """
    Compute retrieval hit-rate at `k`.

    Inputs:
    - `results`: iterable of normalized retrieval-query records.
    - `k`: `int` retrieval depth. Each query is evaluated against the first
      `k` ranked hits. Queries with fewer than `k` hits are counted as misses.

    Returns:
    - `dict[str, object]` containing per-class hit rates plus macro/micro
      aggregates under the requested metric name.

    Example:
        ```python
        payload = compute_hit_at_k(results, k=5)
        ```
    """

    if k <= 0:
        raise ValueError(f"Expected k > 0 for hit@k. Got {k}.")

    normalized_results = list(results)
    per_class_hits: dict[str, int] = defaultdict(int)
    per_class_totals: dict[str, int] = defaultdict(int)
    insufficient_k_queries = 0

    for result in normalized_results:
        if result.query_label is None:
            continue

        per_class_totals[result.query_label] += 1

        top_hits = result.hits[:k]
        if result.available_k < k:
            insufficient_k_queries += 1
            continue

        if any(hit.label == result.query_label for hit in top_hits):
            per_class_hits[result.query_label] += 1

    per_class = {
        label: (
            float(per_class_hits[label]) / float(total)
            if total > 0
            else 0.0
        )
        for label, total in per_class_totals.items()
    }
    macro = float(np.mean(list(per_class.values()))) if per_class else 0.0
    total_queries = sum(per_class_totals.values())
    micro = (
        float(sum(per_class_hits.values())) / float(total_queries)
        if total_queries > 0
        else 0.0
    )

    metric_name = f"hit_at_{k}"
    return {
        metric_name: {
            "k": k,
            "per_class": per_class,
            "macro": macro,
            "micro": micro,
            "num_queries": total_queries,
            "insufficient_k_queries": insufficient_k_queries,
        }
    }

