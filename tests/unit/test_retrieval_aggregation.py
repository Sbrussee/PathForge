from __future__ import annotations

import numpy as np
import pytest

from pathforge.slide_retrieval.aggregation import aggregate_slide_representations
from pathforge.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)


def _representation(data: np.ndarray, indices: list[int]) -> RetrievalRepresentation:
    return RetrievalRepresentation(
        sample_id="slide",
        data=data,
        representation_type="patch_vector",
        additional_data={
            "selected_indices": np.asarray(indices, dtype=np.int32),
            "selected_coords": np.asarray(
                [[index, 0] for index in indices], dtype=np.int32
            ),
            "group_ids": np.asarray([0, 1, 2], dtype=np.int32),
        },
    )


def test_aggregate_concatenates_row_aligned_fields_and_provenance() -> None:
    representation = aggregate_slide_representations(
        sample_id="case-1",
        member_slide_ids=["slide-a", "slide-b"],
        slide_representations=[
            _representation(np.asarray([[1.0, 2.0]]), [4]),
            _representation(np.asarray([[3.0, 4.0], [5.0, 6.0]]), [2, 7]),
        ],
    )

    np.testing.assert_array_equal(
        representation.data, np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    )
    assert representation.additional_data["source_slide_id"].tolist() == [
        "slide-a",
        "slide-b",
        "slide-b",
    ]
    assert representation.additional_data["source_local_selected_index"].tolist() == [
        4,
        2,
        7,
    ]
    assert "group_ids" not in representation.additional_data


def test_aggregate_rejects_incompatible_feature_dimensions() -> None:
    with pytest.raises(ValueError, match="feature dimensions"):
        aggregate_slide_representations(
            sample_id="case-1",
            member_slide_ids=["slide-a", "slide-b"],
            slide_representations=[
                _representation(np.ones((1, 2)), [0]),
                _representation(np.ones((1, 3)), [0]),
            ],
        )


def test_aggregate_rejects_single_vector_without_reducer() -> None:
    with pytest.raises(ValueError, match="no declared aggregation policy"):
        aggregate_slide_representations(
            sample_id="case-1",
            member_slide_ids=["slide-a"],
            slide_representations=[
                RetrievalRepresentation(
                    sample_id="slide-a",
                    data=np.ones(4),
                    representation_type="single_vector",
                )
            ],
        )
