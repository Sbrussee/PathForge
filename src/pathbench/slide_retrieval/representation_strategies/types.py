from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pathbench.slide_retrieval.types import RetrievalItemMetadata


@dataclass(slots=True)
class RetrievalRepresentation:
    """Container for one computed retrieval representation."""
    sample_id: str
    representation_type: str
    data: Any
    metadata: RetrievalItemMetadata = field(default_factory=RetrievalItemMetadata)
    additional_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sample_id = str(self.sample_id)
        self.representation_type = str(self.representation_type)
        self.metadata = RetrievalItemMetadata.from_any(self.metadata)
        self.additional_data = dict(self.additional_data or {})
