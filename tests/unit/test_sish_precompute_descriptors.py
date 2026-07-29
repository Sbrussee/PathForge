from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pathforge.core.io.slide_artifacts import tiles as tiles_io
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.core.io.slide_retrieval import descriptors as descriptors_io
from pathforge.slide_retrieval.representation_strategies.mean_rgb import (
    _slide_retrieval_artifact_path,
)
from pathforge.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathforge.slide_retrieval.search_strategies.strategies.sish.sish_precompute import (
    SISHPrecompute,
)
from pathforge.slide_retrieval.search_strategies.strategies.sish.sish_crops import (
    sish_descriptor_contract,
)


def test_sish_precompute_uses_selected_rows_from_retrieval_descriptor_h5(
    tmp_path,
) -> None:
    artifact_path = tmp_path / "slide-1.h5"
    retrieval_artifact_path = _slide_retrieval_artifact_path(
        slide_artifact_path=artifact_path,
        slide_id="slide-1",
    )
    bag_id = "256px_0.5mpp"
    descriptor_name = "sish_vqvae_latent"

    with FileHandleH5(artifact_path, mode="a") as slide_artifact:
        tiles_io.write_coords(
            slide_artifact,
            bag_id,
            np.asarray(
                [
                    [0, 0, 256, 256, 0],
                    [10, 10, 256, 256, 0],
                    [20, 20, 256, 256, 0],
                    [30, 30, 256, 256, 0],
                ],
                dtype=np.int32,
            ),
        )
        tiles_io.write_tiling_spec(
            slide_artifact,
            bag_id,
            {
                "tile_px": 256,
                "tile_mpp": 0.5,
                "stride_px": 256,
                "coord_space": "level0",
            },
        )

    retrieval_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with FileHandleH5(retrieval_artifact_path, mode="a") as retrieval_artifact:
        descriptors_io.write_descriptor(
            retrieval_artifact,
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
            metadata=sish_descriptor_contract(),
        )
        assert not descriptors_io.descriptor_exists(
            retrieval_artifact,
            bag_id,
            descriptor_name,
            expected_rows=4,
            expected_metadata={"crop_px": 512},
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
        np.asarray([[32], [96]], dtype=np.uint8),
    )


def test_sish_precompute_rejects_descriptor_dim_that_conflicts_with_config(
    tmp_path,
) -> None:
    artifact_path = tmp_path / "slide-1.h5"
    retrieval_artifact_path = _slide_retrieval_artifact_path(
        slide_artifact_path=artifact_path,
        slide_id="slide-1",
    )
    bag_id = "256px_0.5mpp"
    descriptor_name = "sish_vqvae_latent"
    coords = np.asarray(
        [
            [0, 0, 256, 256, 0],
            [10, 10, 256, 256, 0],
            [20, 20, 256, 256, 0],
            [30, 30, 256, 256, 0],
        ],
        dtype=np.int32,
    )
    with FileHandleH5(artifact_path, mode="a") as slide_artifact:
        tiles_io.write_coords(slide_artifact, bag_id, coords)
        tiles_io.write_tiling_spec(
            slide_artifact,
            bag_id,
            {
                "tile_px": 256,
                "tile_mpp": 0.5,
                "stride_px": 256,
                "coord_space": "level0",
            },
        )
    retrieval_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with FileHandleH5(retrieval_artifact_path, mode="a") as retrieval_artifact:
        descriptors_io.write_descriptor(
            retrieval_artifact,
            bag_id,
            descriptor_name,
            np.ones((4, 4), dtype=np.float32),
            metadata=sish_descriptor_contract(),
        )

    representation = RetrievalRepresentation(
        sample_id="slide-1",
        data=np.ones((2, 4), dtype=np.float32),
        additional_data={
            "selected_indices": np.asarray([1, 3], dtype=np.int64),
            "selected_coords": np.asarray([[10, 10], [30, 30]], dtype=np.int32),
        },
    )
    precompute = SISHPrecompute(
        config=SimpleNamespace(
            sish=SimpleNamespace(descriptor_name=descriptor_name, descriptor_dim=16)
        )
    )
    precompute._load_sample_full_coords = lambda sample, bag_id: coords

    with pytest.raises(ValueError, match="expected configured descriptor_dim=16"):
        precompute.enrich_representation(
            representation=representation,
            sample=SimpleNamespace(
                slide_ids=["slide-1"], artifact_paths=[artifact_path]
            ),
            bag_id=bag_id,
        )
