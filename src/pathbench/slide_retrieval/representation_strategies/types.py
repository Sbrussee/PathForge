from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass(slots=True)
class RetrievalRepresentation:
    """Container for one computed retrieval representation."""

    sample_id: str
    data: Any
    exclusion_key: str | None = None
    additional_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sample_id = str(self.sample_id)
        self.exclusion_key = (
            None if self.exclusion_key is None else str(self.exclusion_key)
        )
        self.additional_data = dict(self.additional_data or {})
