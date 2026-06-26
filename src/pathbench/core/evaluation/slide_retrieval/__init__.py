"""Slide-retrieval evaluation package."""

from pathbench.core.evaluation.slide_retrieval.adapter import (
    SlideRetrievalEvaluationAdapter,
)
from pathbench.core.evaluation.slide_retrieval.data import (
    SlideRetrievalEvaluationData,
    SlideRetrievalEvaluationHit,
    SlideRetrievalEvaluationQuery,
)

__all__ = [
    "SlideRetrievalEvaluationAdapter",
    "SlideRetrievalEvaluationData",
    "SlideRetrievalEvaluationHit",
    "SlideRetrievalEvaluationQuery",
]
