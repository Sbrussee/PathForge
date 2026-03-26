from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


@dataclass(slots=True)
class RetrievalItemMetadata(Mapping[str, Any]):
    """
    Shared metadata container for slide retrieval objects.

    This is intentionally Mapping-like so existing code can keep using:
        metadata.get("patient_id")
        dict(metadata)

    Standard fields:
    - category: human-readable class / diagnosis / label
    - patient_id: patient identifier used for filtering
    - case_id: optional case identifier
    - member_ids: members of the aggregated item used for eval lookup output
    - center_id: optional center identifier
    - extra: anything else that should travel with the retrieval item
    """

    category: str | None = None
    patient_id: str | None = None
    case_id: str | None = None
    member_ids: list[str] = field(default_factory=list)
    center_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.member_ids = [str(member) for member in self.member_ids]
        self.extra = dict(self.extra or {})

    def __getitem__(self, key: str) -> Any:
        data = self.to_dict()

        if key in data:
            return data[key]

        if key in {"label", "diagnosis", "target", "class_name"}:
            return self.category
        if key in {"slide_ids", "members", "instance_ids"}:
            return list(self.member_ids)

        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        data = {
            "category": self.category,
            "patient_id": self.patient_id,
            "case_id": self.case_id,
            "member_ids": list(self.member_ids),
            "center_id": self.center_id,
        }
        data.update(self.extra)
        return data

    def copy(self, **updates: Any) -> RetrievalItemMetadata:
        data = self.to_dict()
        data.update(updates)
        return RetrievalItemMetadata.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> RetrievalItemMetadata:
        if data is None:
            return cls()

        data = dict(data)

        member_ids = (
            data.pop("member_ids", None)
            or data.pop("slide_ids", None)
            or data.pop("members", None)
            or data.pop("instance_ids", None)
            or []
        )

        category = (
            data.pop("category", None)
            or data.pop("label", None)
            or data.pop("diagnosis", None)
            or data.pop("target", None)
            or data.pop("class_name", None)
        )

        return cls(
            category=category,
            patient_id=data.pop("patient_id", None),
            case_id=data.pop("case_id", None),
            member_ids=list(member_ids),
            center_id=data.pop("center_id", None),
            extra=dict(data),
        )

    @classmethod
    def from_any(cls, value: Any) -> RetrievalItemMetadata:
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls.from_dict(value)
        if value is None:
            return cls()
        raise TypeError(
            f"Cannot convert metadata of type {type(value).__name__} "
            "to RetrievalItemMetadata."
        )


@dataclass(frozen=True, slots=True)
class SlideRetrievalRunSpec:
    """Immutable identity for one slide-retrieval run."""

    project_root: Path
    bag_id: str
    aggregation_level: str
    feature_extraction: str
    slide_representation: str
    search_method: str
    representation_id: str
    exclude_same_patient: bool = True
    slide_representation_params: dict[str, Any] = field(default_factory=dict)
    search_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", Path(self.project_root))
        object.__setattr__(
            self,
            "slide_representation_params",
            dict(self.slide_representation_params),
        )
        object.__setattr__(self, "search_params", dict(self.search_params))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": "slide_retrieval",
            "bag_id": self.bag_id,
            "aggregation_level": self.aggregation_level,
            "feature_extraction": self.feature_extraction,
            "slide_representation": self.slide_representation,
            "slide_representation_params": dict(self.slide_representation_params),
            "search_method": self.search_method,
            "search_params": dict(self.search_params),
            "representation_id": self.representation_id,
            "exclude_same_patient": self.exclude_same_patient,
        }

    def build_output_dir(self) -> Path:
        bag_component = (
            f"{_safe_name(self.bag_id)}_{_safe_name(self.feature_extraction)}"
        )
        method_component = (
            f"{_safe_name(self.slide_representation)}_"
            f"{_safe_name(self.search_method, hyphenate_underscores=True)}"
        )
        return (
            self.project_root
            / "eval"
            / "slide_retrieval"
            / bag_component
            / method_component
            / f"run_{self.short_hash()}"
        )

    def short_hash(self, length: int = 8) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True, slots=True)
class SlideRetrievalOutputPaths:
    """Resolved filesystem paths for one retrieval run."""

    output_dir: Path
    manifest_path: Path
    metrics_csv_path: Path
    query_results_csv_path: Path
    aggregation_membership_csv_path: Path

    @classmethod
    def from_run_spec(
        cls,
        run_spec: SlideRetrievalRunSpec,
    ) -> SlideRetrievalOutputPaths:
        output_dir = run_spec.build_output_dir()
        return cls(
            output_dir=output_dir,
            manifest_path=output_dir / "manifest.json",
            metrics_csv_path=output_dir / "metrics.csv",
            query_results_csv_path=output_dir / "query_results.csv",
            aggregation_membership_csv_path=output_dir / "aggregation_membership.csv",
        )


@dataclass(frozen=True, slots=True)
class SlideRetrievalManifest:
    """Serializable manifest payload for one retrieval run."""

    run_spec: SlideRetrievalRunSpec
    num_queries: int
    num_reference_items: int
    top_k_saved: int

    def to_dict(self) -> dict[str, Any]:
        manifest = self.run_spec.to_dict()
        manifest.update(
            {
                "num_queries": self.num_queries,
                "num_reference_items": self.num_reference_items,
                "top_k_saved": self.top_k_saved,
            }
        )
        return manifest


def _safe_name(value: Any, *, hyphenate_underscores: bool = False) -> str:
    text = str(value).strip()
    if hyphenate_underscores:
        text = text.replace("_", "-")
    text = text.replace(" ", "-")
    text = text.replace("/", "-")
    text = text.replace("\\", "-")
    while "--" in text:
        text = text.replace("--", "-")
    allowed = []
    for character in text:
        if character.isalnum() or character in "._-":
            allowed.append(character)
        else:
            allowed.append("-")
    return "".join(allowed).strip("-")
