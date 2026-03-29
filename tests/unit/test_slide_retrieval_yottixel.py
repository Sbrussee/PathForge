from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pathbench.core.datasets.bag_dataset import BagSample
from pathbench.core.io.h5 import descriptors as descriptors_io
from pathbench.core.io.h5 import tiles as tiles_io
from pathbench.core.io.h5.base import FileHandleH5
from pathbench.slide_retrieval.representation_strategies.strategies.yottixel import (
    YottixelFeatures,
    YottixelRGB,
)


def _write_coords_artifact(
    tmp_path: Path,
    *,
    artifact_name: str,
    bag_id: str,
    coords_xy: np.ndarray,
    mean_rgb: np.ndarray | None = None,
) -> Path:
    """
    Create one temporary slide artifact with row-aligned coords.

    Args:
        tmp_path: Temporary pytest directory.
        artifact_name: Output H5 filename stem.
        bag_id: Tiling identifier used in the H5 path layout.
        coords_xy: Coordinate array with shape `(N, 2)`.

    Returns:
        Path: Artifact path containing a `(N, 5)` coords dataset.

    Example:
        >>> import numpy as np
        >>> from pathlib import Path
        >>> path = _write_coords_artifact(
        ...     Path("."),
        ...     artifact_name="sample.h5",
        ...     bag_id="256px_0.5mpp",
        ...     coords_xy=np.array([[0, 0]], dtype=np.int32),
        ... )
    """
    artifact_path = tmp_path / artifact_name
    coords_xy = np.asarray(coords_xy, dtype=np.int32)
    coords = np.zeros((coords_xy.shape[0], 5), dtype=np.int32)
    coords[:, :2] = coords_xy
    coords[:, 2] = 256
    coords[:, 3] = 256

    with FileHandleH5(artifact_path, mode="a") as slide_artifact:
        tiles_io.write_coords(slide_artifact, bag_id=bag_id, coords=coords)
        tiles_io.write_tiling_spec(
            slide_artifact,
            bag_id=bag_id,
            tiling_spec={
                "tile_px": 256,
                "tile_mpp": 0.5,
                "stride_px": 256,
                "coord_space": "level0",
                "backend": "lazyslide",
            },
        )
        if mean_rgb is not None:
            descriptors_io.write_descriptor(
                slide_artifact,
                bag_id,
                "mean_rgb",
                mean_rgb,
            )

    return artifact_path


def _make_sample(artifact_path: Path) -> BagSample:
    """
    Build a minimal retrieval sample for one artifact-backed slide.

    Args:
        artifact_path: Path to the temporary slide artifact.

    Returns:
        BagSample: Sample with one slide id and one artifact path.

    Example:
        >>> sample = _make_sample(Path("slide.h5"))
        >>> sample.sample_id
        'sample-1'
    """
    return BagSample(
        sample_id="sample-1",
        slide_ids=["slide-1"],
        artifact_paths=[artifact_path],
        category="tumor",
        metadata={"dataset": "dataset-a"},
    )


def test_yottixel_features_selects_first_member_when_one_rep_per_cluster(
    tmp_path: Path,
) -> None:
    bag_id = "256px_0.5mpp"
    artifact_path = _write_coords_artifact(
        tmp_path,
        artifact_name="slide.h5",
        bag_id=bag_id,
        coords_xy=np.array(
            [
                [0, 0],
                [1, 0],
                [100, 100],
                [101, 100],
            ],
            dtype=np.int32,
        ),
    )
    sample = _make_sample(artifact_path)
    combo_cfg = SimpleNamespace(tile_px=256, tile_mpp=0.5)
    bag = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.0, 10.1],
        ],
        dtype=torch.float32,
    )

    strategy = YottixelFeatures(
        params={"n_clusters": 2, "perc_selected": 50.0},
        config=SimpleNamespace(experiment=SimpleNamespace(random_state=0)),
    )
    representation = strategy.run(bag=bag, sample=sample, combo_cfg=combo_cfg)

    assert representation.sample_id == "sample-1"
    assert representation.representation_type == "patch_vector"
    np.testing.assert_array_equal(
        representation.additional_data["selected_indices"],
        np.array([0, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        representation.data,
        bag.numpy()[[0, 2]],
    )
    assert representation.additional_data["group_ids"].shape == (4,)
    np.testing.assert_array_equal(
        representation.additional_data["selected_coords"],
        np.array(
            [
                [0, 0],
                [100, 100],
            ],
            dtype=np.int32,
        ),
    )


def test_yottixel_features_rejects_mismatched_coords_rows(tmp_path: Path) -> None:
    bag_id = "256px_0.5mpp"
    artifact_path = _write_coords_artifact(
        tmp_path,
        artifact_name="slide.h5",
        bag_id=bag_id,
        coords_xy=np.array([[0, 0]], dtype=np.int32),
    )
    sample = _make_sample(artifact_path)
    combo_cfg = SimpleNamespace(tile_px=256, tile_mpp=0.5)
    bag = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    strategy = YottixelFeatures(params={"n_clusters": 2, "perc_selected": 50.0})

    with pytest.raises(ValueError, match="Bag rows and coordinate rows must match"):
        strategy.run(bag=bag, sample=sample, combo_cfg=combo_cfg)


def test_yottixel_rgb_returns_selected_patch_rows_and_auxiliary_arrays(
    tmp_path: Path,
) -> None:
    bag_id = "256px_0.5mpp"
    artifact_path = _write_coords_artifact(
        tmp_path,
        artifact_name="slide.h5",
        bag_id=bag_id,
        coords_xy=np.array([[0, 0], [100, 100]], dtype=np.int32),
        mean_rgb=np.array([[0.2, 0.8, 0.1], [0.9, 0.1, 0.4]], dtype=np.float32),
    )
    sample = _make_sample(artifact_path)
    combo_cfg = SimpleNamespace(tile_px=256, tile_mpp=0.5)
    bag = torch.tensor([[9.0, 9.0], [8.0, 8.0]], dtype=torch.float32)

    strategy = YottixelRGB(
        params={"n_clusters": 9, "perc_selected": 50.0},
        config=SimpleNamespace(
            slide_processing=SimpleNamespace(backend="lazyslide"),
            datasets=[SimpleNamespace(name="dataset-a", slides_dir=str(tmp_path))],
        ),
    )
    representation = strategy.run(bag=bag, sample=sample, combo_cfg=combo_cfg)

    assert representation.sample_id == "sample-1"
    assert representation.representation_type == "patch_vector"
    np.testing.assert_array_equal(
        representation.additional_data["selected_indices"],
        np.array([0, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        representation.data,
        np.array([[0.2, 0.8, 0.1], [0.9, 0.1, 0.4]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        representation.additional_data["selected_coords"],
        np.array([[0, 0], [100, 100]], dtype=np.int32),
    )
    assert representation.additional_data["group_ids"].shape == (2,)


def test_yottixel_rgb_handles_empty_patch_bag(tmp_path: Path) -> None:
    bag_id = "256px_0.5mpp"
    artifact_path = _write_coords_artifact(
        tmp_path,
        artifact_name="slide.h5",
        bag_id=bag_id,
        coords_xy=np.empty((0, 2), dtype=np.int32),
        mean_rgb=np.empty((0, 3), dtype=np.float32),
    )
    sample = _make_sample(artifact_path)
    combo_cfg = SimpleNamespace(tile_px=256, tile_mpp=0.5)
    bag = torch.empty((0, 3), dtype=torch.float32)

    strategy = YottixelRGB(
        params={"n_clusters": 3, "perc_selected": 50.0},
        config=SimpleNamespace(
            slide_processing=SimpleNamespace(backend="lazyslide"),
            datasets=[SimpleNamespace(name="dataset-a", slides_dir=str(tmp_path))],
        ),
    )
    representation = strategy.run(bag=bag, sample=sample, combo_cfg=combo_cfg)

    assert representation.data.shape == (0, 3)
    assert representation.additional_data["selected_indices"].shape == (0,)
    assert representation.additional_data["group_ids"].shape == (0,)
    assert representation.additional_data["selected_coords"].shape == (0, 2)
