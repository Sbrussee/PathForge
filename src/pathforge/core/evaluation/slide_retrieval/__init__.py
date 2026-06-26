"""Slide-retrieval evaluation package."""

from pathforge.core.evaluation.slide_retrieval.adapter import (
    SlideRetrievalEvaluationAdapter,
)
from pathforge.core.evaluation.slide_retrieval.data import (
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
