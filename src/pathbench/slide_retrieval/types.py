from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


ExclusionLevel = Literal["none", "slide", "case", "patient"]


@dataclass(frozen=True, slots=True)
class RetrievalItemIdentity:
    """Minimal runtime identity for one retrieval item."""

    sample_id: str
    exclusion_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", str(self.sample_id))
        object.__setattr__(
            self,
            "exclusion_key",
            None if self.exclusion_key is None else str(self.exclusion_key),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "exclusion_key": self.exclusion_key,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None,
        *,
        fallback_sample_id: str | None = None,
    ) -> RetrievalItemIdentity:
        payload = dict(data or {})
        sample_id = payload.get("sample_id") or fallback_sample_id
        if not sample_id:
            raise ValueError("Retrieval identity requires a non-empty sample_id.")

        return cls(
            sample_id=str(sample_id),
            exclusion_key=payload.get("exclusion_key"),
        )


@dataclass(frozen=True, slots=True)
class SlideRetrievalManifest:
    """Save-only manifest for one retrieval run."""

    tiling_id: str
    aggregation_level: str
    feature_extraction: str
    slide_representation: str
    search_method: str
    representation_id: str
    exclusion_level: ExclusionLevel
    num_queries: int
    num_reference_items: int
    top_k_saved: int
    slide_representation_params: dict[str, Any] = field(default_factory=dict)
    search_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tiling_id": self.tiling_id,
            "aggregation_level": self.aggregation_level,
            "feature_extraction": self.feature_extraction,
            "slide_representation": self.slide_representation,
            "slide_representation_params": dict(self.slide_representation_params),
            "search_method": self.search_method,
            "search_params": dict(self.search_params),
            "representation_id": self.representation_id,
            "exclusion_level": self.exclusion_level,
            "num_queries": self.num_queries,
            "num_reference_items": self.num_reference_items,
            "top_k_saved": self.top_k_saved,
        }

    def short_hash(self, length: int = 8) -> str:
        serialized = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:length]
