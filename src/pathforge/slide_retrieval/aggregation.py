"""Transient aggregation of cached, physical-slide retrieval representations."""

from __future__ import annotations

from typing import Any

import numpy as np

from pathforge.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathforge.slide_retrieval.types import RetrievalItemMetadata

_CONCATENATABLE_KINDS = frozenset({"patch_vector", "multi_vector"})


def aggregate_slide_representations(
    *,
    sample_id: str,
    member_slide_ids: list[str],
    slide_representations: list[RetrievalRepresentation],
    category: Any = None,
    patient_id: str | None = None,
    case_id: str | None = None,
) -> RetrievalRepresentation:
    """Concatenate cached physical-slide representations for one retrieval item.

    Inputs are ordered identically to ``member_slide_ids``.  Only arrays whose
    first dimension equals the selected-vector row count are concatenated;
    other strategy diagnostics remain slide-local and are intentionally not
    invented at case/patient level.
    """
    if not slide_representations:
        raise ValueError("Cannot aggregate an empty list of slide representations.")
    if len(member_slide_ids) != len(slide_representations):
        raise ValueError(
            "member_slide_ids and slide_representations must have equal length."
        )

    kinds = {
        representation.representation_type for representation in slide_representations
    }
    if len(kinds) != 1:
        raise ValueError(
            f"Cannot aggregate incompatible representation kinds: {sorted(kinds)}."
        )
    kind = kinds.pop()
    if kind not in _CONCATENATABLE_KINDS:
        raise ValueError(
            f"Representation kind {kind!r} has no declared aggregation policy; "
            "only patch_vector and multi_vector can be concatenated."
        )

    arrays = [
        np.asarray(representation.data) for representation in slide_representations
    ]
    if any(array.ndim != 2 for array in arrays):
        raise ValueError(f"{kind} representations must have shape (N, D) to aggregate.")
    dimensions = {int(array.shape[1]) for array in arrays}
    if len(dimensions) != 1:
        raise ValueError(f"Cannot aggregate feature dimensions: {sorted(dimensions)}.")

    data = np.concatenate(arrays, axis=0)
    row_aligned_keys = set.intersection(
        *(
            set(representation.additional_data)
            for representation in slide_representations
        )
    )
    additional_data: dict[str, Any] = {}
    for key in sorted(row_aligned_keys):
        values = [
            np.asarray(representation.additional_data[key])
            for representation in slide_representations
        ]
        if all(
            value.ndim >= 1 and value.shape[0] == array.shape[0]
            for value, array in zip(values, arrays)
        ):
            additional_data[key] = np.concatenate(values, axis=0)

    local_indices = additional_data.get("selected_indices")
    if local_indices is None:
        local_indices = np.concatenate(
            [np.arange(array.shape[0], dtype=np.int32) for array in arrays]
        )
    string_width = max(len(str(slide_id)) for slide_id in member_slide_ids)
    additional_data["source_slide_id"] = np.concatenate(
        [
            np.full(array.shape[0], str(slide_id), dtype=f"<U{string_width}")
            for slide_id, array in zip(member_slide_ids, arrays)
        ]
    )
    additional_data["source_local_selected_index"] = np.asarray(
        local_indices, dtype=np.int32
    )
    additional_data["member_slide_ids"] = np.asarray(member_slide_ids, dtype=str)

    return RetrievalRepresentation(
        sample_id=sample_id,
        data=data,
        representation_type=kind,
        metadata=RetrievalItemMetadata(
            category=category,
            patient_id=patient_id,
            case_id=case_id,
            member_ids=list(member_slide_ids),
        ),
        additional_data=additional_data,
    )
