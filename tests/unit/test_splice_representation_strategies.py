from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import pytest

from pathforge.core.datasets.bag_dataset import BagSample
from pathforge.core.io.slide_artifacts import tiles as tiles_io
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.slide_retrieval.representation_strategies.registry import (
    build_representation_strategy,
    import_representation_strategy_modules,
    is_representation_strategy_available,
)
from pathforge.slide_retrieval.representation_strategies.strategies.splice import (
    SPLICEFeatures,
    SPLICERGB,
)


def _write_coords_artifact(
    artifact_path: Path,
    *,
    bag_id: str,
    coords: np.ndarray,
    mean_rgb: np.ndarray | None = None,
) -> None:
    """
    Write one minimal H5 artifact for SPLICE strategy tests.

    Inputs:
    - artifact_path: `Path` to the temporary H5 file.
    - bag_id: `str` identifying the current tiling group.
    - coords: `np.ndarray` with shape `(N, 5)` and dtype compatible with `int32`.

    Returns:
    - `None`. The function creates or overwrites the coordinate dataset in-place.

    Example:
    ```python
    _write_coords_artifact(path, bag_id="256px_0.5mpp", coords=coords)
    ```
    """
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


def _build_sample(artifact_path: Path) -> BagSample:
    """
    Build a minimal `BagSample` used by the SPLICE tests.

    Inputs:
    - artifact_path: `Path` to the temporary H5 file containing patch coordinates.

    Returns:
    - `BagSample` with one slide member and one artifact path.

    Example:
    ```python
    sample = _build_sample(artifact_path)
    ```
    """
    return BagSample(
        sample_id="sample-1",
        slide_ids=["slide-1"],
        artifact_paths=[artifact_path],
        category="tumor",
        patient_id="patient-1",
        case_id="case-1",
        metadata={"dataset": "dataset-a"},
    )


def _build_combo_cfg() -> SimpleNamespace:
    """
    Build the minimal combo config needed by the SPLICE strategies.

    Inputs:
    - None.

    Returns:
    - `SimpleNamespace` exposing `tile_px: int`, `tile_mpp: float`, and
      `feature_extraction: str`.

    Example:
    ```python
    combo_cfg = _build_combo_cfg()
    ```
    """
    return SimpleNamespace(
        tile_px=256,
        tile_mpp=0.5,
        feature_extraction="uni",
    )


def test_splice_features_selects_expected_rows_and_groups(tmp_path: Path) -> None:
    coords = np.array(
        [
            [10, 11, 256, 256, 0],
            [20, 21, 256, 256, 0],
            [30, 31, 256, 256, 0],
        ],
        dtype=np.int32,
    )
    artifact_path = tmp_path / "slide-1.h5"
    _write_coords_artifact(artifact_path, bag_id="256px_0.5mpp", coords=coords)

    bag = torch.tensor(
        [
            [0.0, 0.0],
            [0.01, 0.0],
            [5.0, 0.0],
        ],
        dtype=torch.float32,
    )
    strategy = SPLICEFeatures(params={"percentile_threshold": 75.0})

    representation = strategy.run(
        bag=bag,
        sample=_build_sample(artifact_path),
        combo_cfg=_build_combo_cfg(),
        coords=coords[:, :2],
        tiling_id="256px_0.5mpp",
    )

    np.testing.assert_allclose(
        representation.data,
        np.array([[0.0, 0.0], [5.0, 0.0]], dtype=float),
    )
    np.testing.assert_array_equal(
        representation.additional_data["selected_indices"],
        np.array([0, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        representation.additional_data["group_ids"],
        np.array([0, 0, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        representation.additional_data["selected_coords"],
        np.array([[10, 11], [30, 31]], dtype=np.int64),
    )
    assert representation.sample_id == "sample-1"


def test_splice_rgb_uses_same_selection_logic_for_histogram_rows(tmp_path: Path) -> None:
    coords = np.array(
        [
            [100, 101, 256, 256, 0],
            [200, 201, 256, 256, 0],
            [300, 301, 256, 256, 0],
        ],
        dtype=np.int32,
    )
    artifact_path = tmp_path / "slide-1.h5"
    _write_coords_artifact(
        artifact_path,
        bag_id="256px_0.5mpp",
        coords=coords,
        mean_rgb=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    bag = torch.tensor(
        [
            [9.0, 9.0, 9.0],
            [8.0, 8.0, 8.0],
            [7.0, 7.0, 7.0],
        ],
        dtype=torch.float32,
    )
    strategy = SPLICERGB(
        params={"percentile_threshold": 75.0},
        config=SimpleNamespace(
            slide_processing=SimpleNamespace(backend="lazyslide"),
            datasets=[SimpleNamespace(name="dataset-a", slides_dir=str(tmp_path))],
        ),
    )

    representation = strategy.run(
        bag=bag,
        sample=_build_sample(artifact_path),
        combo_cfg=_build_combo_cfg(),
        mean_rgb=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        coords=coords[:, :2],
        tiling_id="256px_0.5mpp",
    )

    np.testing.assert_allclose(
        representation.data,
        np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=float),
    )
    np.testing.assert_array_equal(
        representation.additional_data["selected_indices"],
        np.array([0, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        representation.additional_data["group_ids"],
        np.array([0, 0, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        representation.additional_data["selected_coords"],
        np.array([[100, 101], [300, 301]], dtype=np.int64),
    )


def test_splice_features_empty_bag_returns_empty_representation(tmp_path: Path) -> None:
    artifact_path = tmp_path / "slide-1.h5"
    _write_coords_artifact(
        artifact_path,
        bag_id="256px_0.5mpp",
        coords=np.empty((0, 5), dtype=np.int32),
    )

    strategy = SPLICEFeatures(params={"percentile_threshold": 25.0})
    representation = strategy.run(
        bag=np.empty((0, 8), dtype=np.float32),
        sample=_build_sample(artifact_path),
        combo_cfg=_build_combo_cfg(),
        coords=np.empty((0, 2), dtype=np.int64),
        tiling_id="256px_0.5mpp",
    )

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


def test_splice_features_requires_percentile_threshold(tmp_path: Path) -> None:
    coords = np.array([[1, 2, 256, 256, 0]], dtype=np.int32)
    artifact_path = tmp_path / "slide-1.h5"
    _write_coords_artifact(artifact_path, bag_id="256px_0.5mpp", coords=coords)

    strategy = SPLICEFeatures(params={"percentile_threshold": None})

    with pytest.raises(
        ValueError,
        match="percentile_threshold must be specified for SPLICE",
    ):
        strategy.run(
            bag=np.array([[1.0, 2.0]], dtype=np.float32),
            sample=_build_sample(artifact_path),
            combo_cfg=_build_combo_cfg(),
            coords=np.array([[1, 2]], dtype=np.int64),
            tiling_id="256px_0.5mpp",
        )


def test_splice_strategy_registration_round_trip() -> None:
    import_representation_strategy_modules()

    assert is_representation_strategy_available("splice-features")
    assert is_representation_strategy_available("splice-rgb")
    assert isinstance(
        build_representation_strategy("splice-features"),
        SPLICEFeatures,
    )
    assert isinstance(
        build_representation_strategy("splice-rgb"),
        SPLICERGB,
    )
