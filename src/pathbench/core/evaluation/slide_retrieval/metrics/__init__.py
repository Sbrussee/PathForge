from __future__ import annotations

import numpy as np


def build_label_aggregate_payload(
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
