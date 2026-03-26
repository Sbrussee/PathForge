from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ValidationMetricRequest:
    """
    Parsed validation-metric request for slide retrieval.

    Inputs:
    - `raw_name`: `str` metric request from `experiment.evaluation`, expected
      shape `<metric>_at_<k>`.
    - `metric_name`: `str` metric family name without the `_at_<k>` suffix.
    - `registry_key`: `str` canonical registry key, expected shape
      `<metric>_at_k`.
    - `k`: `int` retrieval depth requested by the metric.

    Returns:
    - Immutable parsed metric request consumed by config validation and task
      execution.

    Example:
        ```python
        request = ValidationMetricRequest(
            raw_name="hit_at_5",
            metric_name="hit",
            registry_key="hit_at_k",
            k=5,
        )
        ```
    """

    raw_name: str
    metric_name: str
    registry_key: str
    k: int


@dataclass(frozen=True, slots=True)
class NormalizedSearchHit:
    """
    Normalized retrieval hit consumed by validation metrics.

    Inputs:
    - `item_id`: `str` identifier of the retrieved reference item.
    - `label`: `str | None` class label derived from retrieval metadata.
    - `patient_id`: `str | None` patient identifier derived from retrieval
      metadata.
    - `score`: `float` search score used for ranking.
    - `rank`: `int` 1-based rank of the hit.

    Returns:
    - Immutable hit record with scalar fields only.

    Example:
        ```python
        hit = NormalizedSearchHit(
            item_id="slide-1",
            label="tumor",
            patient_id="patient-1",
            score=0.91,
            rank=1,
        )
        ```
    """

    item_id: str
    label: str | None
    patient_id: str | None
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class NormalizedSearchResult:
    """
    Normalized retrieval-query record consumed by validation metrics.

    Inputs:
    - `query_id`: `str` query sample identifier.
    - `query_label`: `str | None` query class label derived from retrieval
      metadata.
    - `query_patient_id`: `str | None` query patient identifier.
    - `hits`: `list[NormalizedSearchHit]` ranked retrieval hits.
    - `available_k`: `int` number of ranked hits available for evaluation.

    Returns:
    - Immutable normalized query result that decouples metrics from raw
      strategy-specific output objects.

    Example:
        ```python
        result = NormalizedSearchResult(
            query_id="query-1",
            query_label="tumor",
            query_patient_id="patient-1",
            hits=[],
            available_k=0,
        )
        ```
    """

    query_id: str
    query_label: str | None
    query_patient_id: str | None
    hits: list[NormalizedSearchHit] = field(default_factory=list)
    available_k: int = 0

