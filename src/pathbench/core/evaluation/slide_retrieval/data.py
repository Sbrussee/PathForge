from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SlideRetrievalEvaluationHit:
    """One normalized retrieval hit for evaluation."""

    sample_id: str
    label: str
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class SlideRetrievalEvaluationQuery:
    """One normalized retrieval query with resolved labels."""

    query_id: str
    query_label: str
    hits: list[SlideRetrievalEvaluationHit] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SlideRetrievalEvaluationData:
    """Task-specific evaluation payload for slide retrieval."""

    queries: list[SlideRetrievalEvaluationQuery]
