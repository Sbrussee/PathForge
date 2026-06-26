from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Mapping
from collections.abc import MutableMapping


ExclusionLevel = Literal["none", "slide", "case", "patient"]


@dataclass(slots=True)
class RetrievalItemMetadata(MutableMapping[str, Any]):
    """Backward-compatible metadata container for retrieval items."""

    category: str | None = None
    patient_id: str | None = None
    case_id: str | None = None
    member_ids: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if key in {"category", "patient_id", "case_id", "member_ids"}:
            return getattr(self, key)
        return self.extra[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key in {"category", "patient_id", "case_id", "member_ids"}:
            setattr(self, key, value)
            return
        self.extra[key] = value

    def __delitem__(self, key: str) -> None:
        if key in {"category", "patient_id", "case_id", "member_ids"}:
            setattr(self, key, None if key != "member_ids" else [])
            return
        del self.extra[key]

    def __iter__(self) -> Iterator[str]:
        yield "category"
        yield "patient_id"
        yield "case_id"
        yield "member_ids"
        for key in self.extra:
            yield key

    def __len__(self) -> int:
        return 4 + len(self.extra)


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
        # `exclusion_key` is intentionally runtime-only and must not be
        # persisted in retrieval artifact metadata because exclusion policy can
        # change between runs.
        return {
            "sample_id": self.sample_id,
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

    def build_run_hash(self, length: int = 8) -> str:
        serialized = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:length]
