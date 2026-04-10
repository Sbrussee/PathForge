from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pathbench.core.io.h5.base import FileHandleH5
from pathbench.core.io.h5 import tiles as tiles_io
from pathbench.slide_retrieval.representation_strategies.strategies.sdm_features import (
    SDMFeatures,
)

def test_sdm_features_run_wraps_original_outputs_for_repo_contract(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "sample.h5"
    with FileHandleH5(artifact_path, mode="a") as slide_artifact:
        tiles_io.write_coords(
            slide_artifact,
            bag_id="bag_a",
            coords=np.array(
                [
                    [0, 0, 256, 256, 0],
                    [1, 1, 256, 256, 0],
                    [2, 2, 256, 256, 0],
                ],
                dtype=np.int32,
            ),
        )

    strategy = SDMFeatures(bag_id="bag_a")
    sample = SimpleNamespace(sample_id="sample-1", artifact_paths=[artifact_path])
    combo_cfg = SimpleNamespace(
        tile_px=256,
        tile_mpp=0.5,
        experiment=SimpleNamespace(random_state=0),
    )

    representation = strategy.run(
        bag=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        sample=sample,
        combo_cfg=combo_cfg,
        coords=np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64),
    )

    assert representation.sample_id == "sample-1"
    assert representation.representation_type == "patch_vector"
    np.testing.assert_allclose(
        representation.data,
        np.array(
            [
                [1.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        representation.additional_data["selected_indices"],
        np.array([1, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        representation.additional_data["group_ids"],
        np.array([1, 0, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        representation.additional_data["selected_coords"],
        np.array([[1, 1], [2, 2]], dtype=np.int64),
    )
    assert representation.additional_data["groups"] == {"0": [1], "1": [0, 2]}


def test_sdm_features_run_returns_empty_representation_for_empty_bag(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "sample.h5"
    with FileHandleH5(artifact_path, mode="a") as slide_artifact:
        tiles_io.write_coords(
            slide_artifact,
            bag_id="bag_a",
            coords=np.empty((0, 5), dtype=np.int32),
        )

    strategy = SDMFeatures(bag_id="bag_a")
    sample = SimpleNamespace(sample_id="sample-1", artifact_paths=[artifact_path])
    combo_cfg = SimpleNamespace(tile_px=256, tile_mpp=0.5)

    representation = strategy.run(
        bag=torch.empty((0, 4), dtype=torch.float32),
        sample=sample,
        combo_cfg=combo_cfg,
        coords=np.empty((0, 2), dtype=np.int64),
    )

    assert representation.sample_id == "sample-1"
    assert representation.data.shape == (0, 0)
    np.testing.assert_array_equal(
        representation.additional_data["selected_indices"],
        np.array([], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        representation.additional_data["group_ids"],
        np.array([], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        representation.additional_data["selected_coords"],
        np.empty((0, 2), dtype=np.int64),
    )
    assert representation.additional_data["groups"] == {}


def test_sdm_features_run_rejects_non_finite_features(tmp_path: Path) -> None:
    artifact_path = tmp_path / "sample.h5"
    with FileHandleH5(artifact_path, mode="a") as slide_artifact:
        tiles_io.write_coords(
            slide_artifact,
            bag_id="bag_a",
            coords=np.array(
                [
                    [0, 0, 256, 256, 0],
                    [1, 1, 256, 256, 0],
                ],
                dtype=np.int32,
            ),
        )

    strategy = SDMFeatures(bag_id="bag_a")
    sample = SimpleNamespace(sample_id="sample-1", artifact_paths=[artifact_path])
    combo_cfg = SimpleNamespace(tile_px=256, tile_mpp=0.5)

    with pytest.raises(ValueError, match="Non-finite values in features"):
        strategy.run(
            bag=torch.tensor(
                [
                    [0.0, np.nan],
                    [1.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            sample=sample,
            combo_cfg=combo_cfg,
            coords=np.array([[0, 0], [1, 1]], dtype=np.int64),
        )


def test_sdm_features_run_requires_matching_feature_and_coord_rows(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "sample.h5"
    with FileHandleH5(artifact_path, mode="a") as slide_artifact:
        tiles_io.write_coords(
            slide_artifact,
            bag_id="bag_a",
            coords=np.array([[0, 0, 256, 256, 0]], dtype=np.int32),
        )

    strategy = SDMFeatures(bag_id="bag_a")
    sample = SimpleNamespace(sample_id="sample-1", artifact_paths=[artifact_path])
    combo_cfg = SimpleNamespace(tile_px=256, tile_mpp=0.5)

    with pytest.raises(ValueError, match="one coordinate row per patch feature row"):
        strategy.run(
            bag=torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
            sample=sample,
            combo_cfg=combo_cfg,
            coords=np.array([[0, 0]], dtype=np.int64),
        )
