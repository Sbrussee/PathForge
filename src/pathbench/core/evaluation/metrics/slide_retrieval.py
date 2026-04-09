from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from pathbench.core.evaluation.registry import evaluation_metric
from pathbench.core.evaluation.tasks.slide_retrieval import (
    SlideRetrievalEvaluationData,
)
from pathbench.core.evaluation.types import MetricRequest


def _build_label_aggregate_payload(
    *,
    per_label_values: dict[str, float],
    counts_per_label: dict[str, int],
    evaluable_queries: int,
) -> dict[str, object]:
    """Build the shared retrieval-metric payload shape."""

    ordered_per_label = {
        label: float(value)
        for label, value in sorted(per_label_values.items())
    }
    macro = (
        float(np.mean(list(ordered_per_label.values())))
        if ordered_per_label
        else 0.0
    )
    total_queries = int(sum(counts_per_label.values()))
    weighted_numerator = sum(
        float(ordered_per_label[label]) * float(count)
        for label, count in counts_per_label.items()
    )
    micro = (
        weighted_numerator / float(total_queries)
        if total_queries > 0
        else 0.0
    )

    return {
        "macro": macro,
        "micro": micro,
        "per_label": ordered_per_label,
        "counts": {
            "num_queries": total_queries,
            "num_evaluable_queries": int(evaluable_queries),
            "num_labels": len(ordered_per_label),
        },
        "counts_per_label": dict(sorted(counts_per_label.items())),
    }


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
    """
    Compute slide-retrieval hit-rate at the requested `k`.

    Inputs:
    - `evaluation_data`: normalized slide-retrieval queries and hits.
    - `request`: parsed metric request whose `params` contains integer `k`.
    - `run_context`: unused shared evaluation context accepted for uniformity.

    Returns:
    - JSON-serializable payload with macro, micro, per-label, and count outputs.

    Example:
    ```python
    payload = compute_hit_at_k(
        evaluation_data,
        request=MetricRequest(
            raw_name="hit_at_5",
            canonical_name="hit_at_k",
            params={"k": 5},
        ),
    )
    ```
    """

    _ = run_context

    k = int(request.params["k"])
    if k <= 0:
        raise ValueError(f"Expected k > 0 for hit_at_k. Got {k}.")

    per_label_hits: dict[str, int] = defaultdict(int)
    counts_per_label: dict[str, int] = defaultdict(int)

    for query in evaluation_data.queries:
        counts_per_label[query.query_label] += 1

        top_hits = query.hits[:k]
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
        **_build_label_aggregate_payload(
            per_label_values=per_label,
            counts_per_label=counts_per_label,
            evaluable_queries=sum(counts_per_label.values()),
        ),
    }


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
    """
    Compute majority-vote accuracy at the requested `k`.

    Inputs:
    - `evaluation_data`: normalized slide-retrieval queries and hits.
    - `request`: parsed metric request whose `params` contains integer `k`.
    - `run_context`: unused shared evaluation context accepted for uniformity.

    Returns:
    - JSON-serializable payload with macro, micro, per-label, and count outputs.
    """

    _ = run_context

    k = int(request.params["k"])
    if k <= 0:
        raise ValueError(f"Expected k > 0 for mmv_at_k. Got {k}.")

    correct_counts: dict[str, int] = defaultdict(int)
    counts_per_label: dict[str, int] = defaultdict(int)

    for query in evaluation_data.queries:
        counts_per_label[query.query_label] += 1
        top_hits = query.hits[:k]
        if len(top_hits) < k or not top_hits:
            continue

        majority_label = Counter(hit.label for hit in top_hits).most_common(1)[0][0]
        if majority_label == query.query_label:
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
        **_build_label_aggregate_payload(
            per_label_values=per_label,
            counts_per_label=counts_per_label,
            evaluable_queries=sum(counts_per_label.values()),
        ),
    }


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
    """
    Compute mean average precision at the requested `k`.

    Inputs:
    - `evaluation_data`: normalized slide-retrieval queries and hits.
    - `request`: parsed metric request whose `params` contains integer `k`.
    - `run_context`: unused shared evaluation context accepted for uniformity.

    Returns:
    - JSON-serializable payload with macro, micro, per-label, and count outputs.
    """

    _ = run_context

    k = int(request.params["k"])
    if k <= 0:
        raise ValueError(f"Expected k > 0 for map_at_k. Got {k}.")

    ap_per_label: dict[str, list[float]] = defaultdict(list)
    counts_per_label: dict[str, int] = defaultdict(int)

    for query in evaluation_data.queries:
        counts_per_label[query.query_label] += 1
        top_hits = query.hits[:k]
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
        **_build_label_aggregate_payload(
            per_label_values=per_label,
            counts_per_label=counts_per_label,
            evaluable_queries=sum(counts_per_label.values()),
        ),
    }
