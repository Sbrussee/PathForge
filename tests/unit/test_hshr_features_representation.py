from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pathbench.core.datasets.bag_dataset import BagSample
from pathbench.core.io.h5.base import FileHandleH5
from pathbench.core.io.h5.tiles import write_coords
from pathbench.slide_retrieval.representation_strategies.strategies.hshr_features import (
    HSHRFeatures,
)


class _FakeBagDataset:
    def __init__(self, *, tiling_id: str) -> None:
        self.tiling_id = tiling_id


def _make_sample(*, artifact_path: Path, sample_id: str = "sample-1") -> BagSample:
    return BagSample(
        sample_id=sample_id,
        slide_ids=["slide-1"],
        artifact_paths=[artifact_path],
        category="tumor",
    )


def test_hshr_features_selects_one_feature_per_cluster_and_loads_coords(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "slide-1.h5"
    coords = np.asarray(
        [
            [1, 2, 256, 256, 0],
            [3, 4, 256, 256, 0],
            [5, 6, 256, 256, 0],
            [7, 8, 256, 256, 0],
        ],
        dtype=np.int32,
    )
    with FileHandleH5(artifact_path, mode="a") as slide_artifact:
        write_coords(slide_artifact, bag_id="bag-a", coords=coords)

    bag = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
            [10.0, 11.0],
        ],
        dtype=torch.float32,
    )
    sample = _make_sample(artifact_path=artifact_path)
    strategy = HSHRFeatures(
        params={"n_patches": 2},
        config=SimpleNamespace(experiment=SimpleNamespace(random_state=0)),
    )

    representation = strategy.run(
        bag=bag,
        sample=sample,
        bag_dataset=_FakeBagDataset(tiling_id="bag-a"),
    )

    assert representation.sample_id == "sample-1"
    assert representation.representation_type == "multi_vector"
    assert representation.data.shape == (2, 2)
    np.testing.assert_array_equal(
        np.sort(representation.additional_data["selected_indices"]),
        np.asarray([0, 2], dtype=int),
    )
    np.testing.assert_array_equal(
        representation.additional_data["selected_coords"],
        coords[representation.additional_data["selected_indices"], :2],
    )
    np.testing.assert_array_equal(
        representation.data,
        bag[representation.additional_data["selected_indices"]].numpy(),
    )
    group_ids = representation.additional_data["group_ids"]
    assert group_ids.shape == (4,)
    assert group_ids[0] == group_ids[1]
    assert group_ids[2] == group_ids[3]
    assert group_ids[0] != group_ids[2]


def test_hshr_features_returns_empty_multi_vector_for_empty_bag() -> None:
    strategy = HSHRFeatures(params={"n_patches": 3})

    representation = strategy.run(
        bag=torch.empty((0, 4), dtype=torch.float32),
        sample=None,
        bag_dataset=None,
    )

    assert representation.sample_id == ""
    assert representation.representation_type == "multi_vector"
    assert representation.data.shape == (0, 0)
    assert representation.additional_data["selected_indices"].shape == (0,)
    assert representation.additional_data["group_ids"].shape == (0,)
    assert representation.additional_data["selected_coords"].shape == (0, 2)


def test_hshr_features_rejects_coordinate_mismatch(tmp_path: Path) -> None:
    artifact_path = tmp_path / "slide-1.h5"
    coords = np.asarray(
        [
            [1, 2, 256, 256, 0],
            [3, 4, 256, 256, 0],
        ],
        dtype=np.int32,
    )
    with FileHandleH5(artifact_path, mode="a") as slide_artifact:
        write_coords(slide_artifact, bag_id="bag-a", coords=coords)

    strategy = HSHRFeatures(params={"n_patches": 2})
    sample = _make_sample(artifact_path=artifact_path)

    try:
        strategy.run(
            bag=torch.randn(3, 2),
            sample=sample,
            bag_dataset=_FakeBagDataset(tiling_id="bag-a"),
        )
    except ValueError as exc:
        assert "number of coords does not match number of bag rows" in str(exc)
    else:
        raise AssertionError("Expected coordinate mismatch to raise ValueError.")
