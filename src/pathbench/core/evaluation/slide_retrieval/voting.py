from __future__ import annotations

from collections import Counter

from pathbench.core.evaluation.slide_retrieval.data import (
    SlideRetrievalEvaluationHit,
    SlideRetrievalEvaluationQuery,
)


def filter_query_self_hits(
    query: SlideRetrievalEvaluationQuery,
) -> list[SlideRetrievalEvaluationHit]:
    """Drop defensive self-hits so all metrics evaluate external neighbors only."""

    return [
        hit
        for hit in query.hits
        if str(hit.sample_id) != str(query.query_id)
    ]


def get_top_k_hits(
    query: SlideRetrievalEvaluationQuery,
    *,
    k: int,
) -> list[SlideRetrievalEvaluationHit]:
    """Return the top-k ranked hits after defensive self-hit filtering."""

    return filter_query_self_hits(query)[:k]


def predict_label_from_top_k(
    query: SlideRetrievalEvaluationQuery,
    *,
    k: int,
) -> str | None:
    """
    Convert retrieval hits into one predicted label using strict-majority voting.

    For `k=1`, return the top hit label after self-hit filtering.
    For `k>1`, vote over the available top `min(k, num_hits)` hits and return a
    label only when it has a strict majority within that available neighborhood.
    Ties return `None`.
    """

    top_hits = get_top_k_hits(query, k=k)
    if not top_hits:
        return None
    if k == 1:
        return top_hits[0].label

    label_counts = Counter(hit.label for hit in top_hits)
    effective_k = len(top_hits)
    majority_threshold = (effective_k // 2) + 1
    for hit in top_hits:
        label = hit.label
        if label_counts[label] >= majority_threshold:
            return label
    return None
