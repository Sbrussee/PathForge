from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathbench.slide_retrieval.sish_precompute import SISHPrecompute
from pathbench.slide_retrieval.types import RetrievalItemMetadata


def test_sish_precompute_enriches_representation_with_indices_and_packed_bits(
    monkeypatch,
) -> None:
    representation = RetrievalRepresentation(
        sample_id="slide-1",
        representation_type="multi_vector",
        data=np.asarray(
            [
                [1.0, -1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0, -1.0],
            ],
            dtype=np.float32,
        ),
        metadata=RetrievalItemMetadata(category="tumor", patient_id="patient-1"),
        additional_data={
            "selected_indices": np.asarray([1, 3], dtype=np.int64),
            "selected_coords": np.asarray([[10, 10], [30, 30]], dtype=np.int32),
        },
    )
    precompute = SISHPrecompute(config=SimpleNamespace())

    monkeypatch.setattr(
        precompute,
        "_load_sample_full_coords",
        lambda sample, bag_id: np.asarray(
            [
                [0, 0, 256, 256, 0],
                [10, 10, 256, 256, 0],
                [20, 20, 256, 256, 0],
                [30, 30, 256, 256, 0],
            ],
            dtype=np.int32,
        ),
    )
    monkeypatch.setattr(
        precompute,
        "_build_patch_specs",
        lambda sample, bag_id, full_coords, selected_indices: ["dummy", "dummy"],
    )
    monkeypatch.setattr(
        precompute,
        "_encode_selected_patch_specs",
        lambda patch_specs: np.asarray([101, 202], dtype=np.int64),
    )

    enriched = precompute.enrich_representation(
        representation=representation,
        sample=SimpleNamespace(slide_ids=["slide-1"], artifact_paths=[]),
        bag_id="256px_0.5mpp",
    )

    np.testing.assert_array_equal(
        enriched.additional_data["sish_patch_indices"],
        np.asarray([101, 202], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        enriched.additional_data["sish_packed_bits"],
        np.packbits((np.asarray(representation.data) > 0).astype(np.uint8), axis=1),
    )
