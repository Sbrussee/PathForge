from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pathbench.slide_retrieval.types import RetrievalItemMetadata


@dataclass(slots=True)
class SearchDatabaseItem:
    """Container for one searchable item."""

    sample_id: str
    data: Any
    metadata: RetrievalItemMetadata = field(default_factory=RetrievalItemMetadata)
    exclusion_key: str | None = None
    additional_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sample_id = str(self.sample_id)
        if not isinstance(self.metadata, RetrievalItemMetadata):
            if isinstance(self.metadata, dict):
                self.metadata = RetrievalItemMetadata(**self.metadata)
            else:
                self.metadata = RetrievalItemMetadata()
        self.exclusion_key = (
            None if self.exclusion_key is None else str(self.exclusion_key)
        )
        self.additional_data = dict(self.additional_data or {})

    @property
    def item_id(self) -> str:
        return self.sample_id


@dataclass(slots=True)
class SearchHit:
    """Container for one ranked retrieval result."""

    sample_id: str
    score: float
    rank: int
    metadata: RetrievalItemMetadata = field(default_factory=RetrievalItemMetadata)

    def __post_init__(self) -> None:
        self.sample_id = str(self.sample_id)
        self.score = float(self.score)
        self.rank = int(self.rank)
        if not isinstance(self.metadata, RetrievalItemMetadata):
            if isinstance(self.metadata, dict):
                self.metadata = RetrievalItemMetadata(**self.metadata)
            else:
                self.metadata = RetrievalItemMetadata()

    @property
    def item_id(self) -> str:
        return self.sample_id


@dataclass(slots=True)
class SearchResult:
    """Container for the output of one query."""

    query_sample_id: str
    hits: list[SearchHit]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.query_sample_id = str(self.query_sample_id)
        self.hits = list(self.hits)
        self.metadata = dict(self.metadata or {})

    @property
    def query_id(self) -> str:
        return self.query_sample_id
