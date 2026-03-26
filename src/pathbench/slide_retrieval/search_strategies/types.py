from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pathbench.slide_retrieval.types import RetrievalItemMetadata


@dataclass(slots=True)
class SearchDatabaseItem:
    """Container for one searchable item."""
    item_id: str
    search_type: str
    data: Any
    metadata: RetrievalItemMetadata = field(default_factory=RetrievalItemMetadata)

    def __post_init__(self) -> None:
        self.item_id = str(self.item_id)
        self.search_type = str(self.search_type)
        self.metadata = RetrievalItemMetadata.from_any(self.metadata)


@dataclass(slots=True)
class SearchHit:
    """Container for one ranked retrieval result."""
    item_id: str
    score: float
    rank: int
    metadata: RetrievalItemMetadata = field(default_factory=RetrievalItemMetadata)

    def __post_init__(self) -> None:
        self.item_id = str(self.item_id)
        self.score = float(self.score)
        self.rank = int(self.rank)
        self.metadata = RetrievalItemMetadata.from_any(self.metadata)


@dataclass(slots=True)
class SearchResult:
    """Container for the output of one query."""
    query_id: str
    hits: list[SearchHit]
    metadata: RetrievalItemMetadata = field(default_factory=RetrievalItemMetadata)

    def __post_init__(self) -> None:
        self.query_id = str(self.query_id)
        self.hits = list(self.hits)
        self.metadata = RetrievalItemMetadata.from_any(self.metadata)
