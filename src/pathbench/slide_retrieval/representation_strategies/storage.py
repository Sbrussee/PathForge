from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def build_retrieval_representation_artifact_path(
    artifacts_dir: str | Path,
    aggregation_level: str,
    sample_id: str,
) -> Path:
    """Build the H5 artifact path for one retrieval representation sample."""
    root = Path(artifacts_dir).expanduser().resolve()
    sample_id = str(sample_id).strip()
    aggregation_level = str(aggregation_level).strip()

    if not sample_id:
        raise ValueError("sample_id must be a non-empty string.")
    if "/" in sample_id or "\\" in sample_id:
        raise ValueError(f"sample_id may not contain path separators: {sample_id!r}")
    if not aggregation_level:
        raise ValueError("aggregation_level must be a non-empty string.")
    if "/" in aggregation_level or "\\" in aggregation_level:
        raise ValueError(
            f"aggregation_level may not contain path separators: {aggregation_level!r}"
        )

    return root / "slide_retrieval" / aggregation_level / f"{sample_id}.h5"


def build_retrieval_representation_id(
    feature_extraction: str,
    retrieval_representation: str,
    params: dict[str, Any] | None = None,
) -> str:
    """Build a stable retrieval representation ID."""
    feature_extraction = _normalize_name(feature_extraction, "feature_extraction")
    retrieval_representation = _normalize_name(
        retrieval_representation,
        "retrieval_representation",
    )

    normalized_params = _normalize_for_json(params or {})
    params_payload = json.dumps(
        normalized_params,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    params_hash = hashlib.sha1(params_payload.encode("utf-8")).hexdigest()[:16]

    return f"{feature_extraction}__{retrieval_representation}__{params_hash}"


def build_retrieval_representation_entry_id(
    slide_ids: list[str],
) -> str:
    """Build a stable entry ID from the member slide IDs."""
    normalized_slide_ids = sorted(
        _normalize_name(slide_id, "slide_id") for slide_id in slide_ids
    )

    if not normalized_slide_ids:
        raise ValueError("slide_ids must contain at least one slide ID.")

    payload = "||".join(normalized_slide_ids)
    member_hash = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    return f"members_{member_hash}"


def _normalize_name(value: Any, field_name: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    if "/" in value:
        raise ValueError(f"{field_name} may not contain '/': {value!r}")
    return value


def _normalize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _normalize_for_json(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }

    if isinstance(value, (list, tuple, set)):
        return [_normalize_for_json(item) for item in value]

    return str(value)
