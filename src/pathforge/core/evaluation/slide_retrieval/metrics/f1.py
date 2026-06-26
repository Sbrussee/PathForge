from __future__ import annotations

from collections import defaultdict

import numpy as np

from pathforge.core.evaluation.registry import evaluation_metric
from pathforge.core.evaluation.slide_retrieval.data import (
    SlideRetrievalEvaluationData,
)
from pathforge.core.evaluation.slide_retrieval.voting import predict_label_from_top_k
from pathforge.core.evaluation.types import MetricRequest


@evaluation_metric(
    "macro_f1_at_k",
    tasks=("slide_retrieval",),
    pattern=r"^macro_f1_at_(?P<k>[1-9]\d*)$",
    param_builder=lambda match: {"k": int(match.group("k"))},
)
def compute_macro_f1_at_k(
    evaluation_data: SlideRetrievalEvaluationData,
    *,
    request: MetricRequest,
    run_context: object | None = None,
) -> dict[str, object]:
    """Compute strict-majority multiclass macro-F1 at the requested `k`."""

    _ = run_context

    k = int(request.params["k"])
    if k <= 0:
        raise ValueError(f"Expected k > 0 for macro_f1_at_k. Got {k}.")

    counts_per_label: dict[str, int] = defaultdict(int)
    true_positive_counts: dict[str, int] = defaultdict(int)
    false_positive_counts: dict[str, int] = defaultdict(int)
    false_negative_counts: dict[str, int] = defaultdict(int)

    for query in evaluation_data.queries:
        true_label = query.query_label
        predicted_label = predict_label_from_top_k(query, k=k)
        counts_per_label[true_label] += 1

        if predicted_label == true_label:
            true_positive_counts[true_label] += 1
            continue

        false_negative_counts[true_label] += 1
        if predicted_label is not None:
            false_positive_counts[predicted_label] += 1

    per_label_f1: dict[str, float] = {}
    labels = sorted(counts_per_label)
    for label in labels:
        tp = true_positive_counts[label]
        fp = false_positive_counts[label]
        fn = false_negative_counts[label]

        precision = (
            float(tp) / float(tp + fp)
            if (tp + fp) > 0
            else 0.0
        )
        recall = (
            float(tp) / float(tp + fn)
            if (tp + fn) > 0
            else 0.0
        )
        per_label_f1[label] = (
            (2.0 * precision * recall) / (precision + recall)
            if (precision + recall) > 0.0
            else 0.0
        )

    total_tp = sum(true_positive_counts.values())
    total_fp = sum(false_positive_counts.values())
    total_fn = sum(false_negative_counts.values())
    micro_precision = (
        float(total_tp) / float(total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0.0
    )
    micro_recall = (
        float(total_tp) / float(total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0.0
    )
    micro_f1 = (
        (2.0 * micro_precision * micro_recall) / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0.0
        else 0.0
    )

    return {
        "k": k,
        "macro": (
            float(np.mean(list(per_label_f1.values())))
            if per_label_f1
            else 0.0
        ),
        "micro": micro_f1,
        "per_label": per_label_f1,
        "counts": {
            "num_queries": int(sum(counts_per_label.values())),
            "num_evaluable_queries": int(sum(counts_per_label.values())),
            "num_labels": len(per_label_f1),
        },
        "counts_per_label": dict(sorted(counts_per_label.items())),
    }
