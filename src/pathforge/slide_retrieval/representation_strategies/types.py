from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pathforge.slide_retrieval.types import RetrievalItemMetadata


@dataclass(slots=True)
class RetrievalRepresentation:
    """Container for one computed retrieval representation."""

    sample_id: str
    data: Any
    representation_type: str = "patch_vector"
    metadata: RetrievalItemMetadata = field(default_factory=RetrievalItemMetadata)
    exclusion_key: str | None = None
    additional_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sample_id = str(self.sample_id)
        self.representation_type = str(self.representation_type)
        if not isinstance(self.metadata, RetrievalItemMetadata):
            if isinstance(self.metadata, dict):
                self.metadata = RetrievalItemMetadata(**self.metadata)
            else:
                self.metadata = RetrievalItemMetadata()
        if self.exclusion_key is None and self.metadata.patient_id is not None:
            self.exclusion_key = str(self.metadata.patient_id)
        self.exclusion_key = (
            None if self.exclusion_key is None else str(self.exclusion_key)
        )
        self.additional_data = dict(self.additional_data or {})
