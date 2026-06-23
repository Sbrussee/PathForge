from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pathbench.core.io.slide_artifacts.base import FileHandleH5
from pathbench.core.io.slide_retrieval import descriptors as descriptors_io
from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathbench.slide_retrieval.search_strategies.strategies.sish.sish_precompute import SISHPrecompute


def test_sish_precompute_uses_selected_rows_from_stored_descriptors(tmp_path) -> None:
    artifact_path = tmp_path / "slide-1.h5"
    bag_id = "256px_0.5mpp"
    descriptor_name = "sish_vqvae_latent"

    with FileHandleH5(artifact_path, mode="a") as slide_artifact:
        descriptors_io.write_descriptor(
            slide_artifact,
            bag_id,
            descriptor_name,
            np.asarray(
                [
                    [11.0],
                    [101.0],
                    [22.0],
                    [202.0],
                ],
                dtype=np.float32,
            ),
        )

    representation = RetrievalRepresentation(
        sample_id="slide-1",
        data=np.asarray(
            [
                [1.0, -1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0, -1.0],
            ],
            dtype=np.float32,
        ),
        additional_data={
            "selected_indices": np.asarray([1, 3], dtype=np.int64),
            "selected_coords": np.asarray([[10, 10], [30, 30]], dtype=np.int32),
        },
    )

    precompute = SISHPrecompute(
        config=SimpleNamespace(
            sish=SimpleNamespace(
                descriptor_name=descriptor_name,
            )
        )
    )
    precompute._load_sample_full_coords = lambda sample, bag_id: np.asarray(
        [
            [0, 0, 256, 256, 0],
            [10, 10, 256, 256, 0],
            [20, 20, 256, 256, 0],
            [30, 30, 256, 256, 0],
        ],
        dtype=np.int32,
    )

    enriched = precompute.enrich_representation(
        representation=representation,
        sample=SimpleNamespace(
            slide_ids=["slide-1"],
            artifact_paths=[artifact_path],
        ),
        bag_id=bag_id,
    )

    np.testing.assert_array_equal(
        enriched.additional_data["sish_patch_indices"],
        np.asarray([101, 202], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        enriched.additional_data["sish_packed_bits"],
        np.packbits((np.asarray(representation.data) > 0).astype(np.uint8), axis=1),
    )
