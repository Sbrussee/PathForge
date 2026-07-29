from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pathforge.core.datasets.bag_dataset import BagSample
from pathforge.core.io.slide_artifacts import tiles as tiles_io
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.slide_retrieval.representation_strategies.storage import (
    build_retrieval_representation_id,
)
from pathforge.slide_retrieval.representation_strategies.strategies.yottixel import (
    YottixelFeatures,
    YottixelRGB,
)


def _sort_rows(array: np.ndarray) -> np.ndarray:
    """Return a lexicographically row-sorted copy of a 2D array."""
    order = np.lexsort((array[:, 1], array[:, 0]))
    return np.asarray(array[order])


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
    representation = strategy.run(
        bag=bag.numpy(),
        sample=sample,
        combo_cfg=combo_cfg,
        coords=np.array(
            [
                [0, 0],
                [1, 0],
                [100, 100],
                [101, 100],
            ],
            dtype=np.int32,
        ),
        tiling_id=bag_id,
    )

    assert representation.sample_id == "sample-1"
    np.testing.assert_array_equal(
        np.sort(representation.additional_data["selected_indices"]),
        np.array([0, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.sort(representation.data, axis=0),
        np.sort(bag.numpy()[[0, 2]], axis=0),
    )
    assert representation.additional_data["group_ids"].shape == (4,)
    np.testing.assert_array_equal(
        _sort_rows(representation.additional_data["selected_coords"]),
        _sort_rows(
            np.array(
                [
                    [0, 0],
                    [100, 100],
                ],
                dtype=np.int32,
            )
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
        strategy.run(
            bag=bag.numpy(),
            sample=sample,
            combo_cfg=combo_cfg,
            coords=np.array([[0, 0]], dtype=np.int32),
            tiling_id=bag_id,
        )


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
    representation = strategy.run(
        bag=bag,
        sample=sample,
        combo_cfg=combo_cfg,
        colour_descriptors=np.array(
            [[0.2, 0.8, 0.1], [0.9, 0.1, 0.4]], dtype=np.float32
        ),
        coords=np.array([[0, 0], [100, 100]], dtype=np.int32),
        tiling_id=bag_id,
    )

    assert representation.sample_id == "sample-1"
    np.testing.assert_array_equal(
        np.sort(representation.additional_data["selected_indices"]),
        np.array([0, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.sort(representation.data, axis=0),
        np.sort(bag.numpy(), axis=0),
    )
    np.testing.assert_array_equal(
        _sort_rows(representation.additional_data["selected_coords"]),
        _sort_rows(np.array([[0, 0], [100, 100]], dtype=np.int32)),
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
    representation = strategy.run(
        bag=bag,
        sample=sample,
        combo_cfg=combo_cfg,
        colour_descriptors=np.empty((0, 3), dtype=np.float32),
        coords=np.empty((0, 2), dtype=np.int32),
        tiling_id=bag_id,
    )

    assert representation.data.shape == (0, 3)
    assert representation.additional_data["selected_indices"].shape == (0,)
    assert representation.additional_data["group_ids"].shape == (0,)
    assert representation.additional_data["selected_coords"].shape == (0, 2)


def test_yottixel_rgb_accepts_histograms_for_selection_and_returns_fm_rows(
    tmp_path: Path,
) -> None:
    bag_id = "256px_0.5mpp"
    artifact_path = _write_coords_artifact(
        tmp_path,
        artifact_name="slide.h5",
        bag_id=bag_id,
        coords_xy=np.array([[0, 0], [100, 100]], dtype=np.int32),
    )
    sample = _make_sample(artifact_path)
    fm_bag = np.array([[9.0, 2.0, 1.0], [8.0, 3.0, 4.0]], dtype=np.float32)
    histograms = np.zeros((2, 768), dtype=np.float32)
    histograms[1, -1] = 1.0
    strategy = YottixelRGB(
        params={
            "colour_descriptor": "histogram_rgb",
            "n_clusters": 2,
            "perc_selected": 50.0,
        },
        config=SimpleNamespace(experiment=SimpleNamespace(random_state=0)),
    )

    representation = strategy.run(
        bag=fm_bag,
        sample=sample,
        combo_cfg=SimpleNamespace(tile_px=256, tile_mpp=0.5),
        colour_descriptors=histograms,
        coords=np.array([[0, 0], [100, 100]], dtype=np.int32),
        tiling_id=bag_id,
    )

    assert strategy.hyperparam_values()["colour_descriptor"] == "histogram_rgb"
    np.testing.assert_array_equal(_sort_rows(representation.data), _sort_rows(fm_bag))


def test_yottixel_rgb_rejects_misaligned_fm_and_colour_rows(tmp_path: Path) -> None:
    bag_id = "256px_0.5mpp"
    artifact_path = _write_coords_artifact(
        tmp_path,
        artifact_name="slide.h5",
        bag_id=bag_id,
        coords_xy=np.array([[0, 0], [100, 100]], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="Foundation-model bag rows"):
        YottixelRGB().run(
            bag=np.ones((1, 4), dtype=np.float32),
            sample=_make_sample(artifact_path),
            combo_cfg=SimpleNamespace(tile_px=256, tile_mpp=0.5),
            colour_descriptors=np.ones((2, 3), dtype=np.float32),
            coords=np.array([[0, 0], [100, 100]], dtype=np.int32),
            tiling_id=bag_id,
        )


def test_yottixel_rgb_selection_uses_colour_rows_and_returns_matching_fm_rows(
    tmp_path: Path,
) -> None:
    bag_id = "256px_0.5mpp"
    artifact_path = _write_coords_artifact(
        tmp_path,
        artifact_name="slide.h5",
        bag_id=bag_id,
        coords_xy=np.array([[0, 0], [100, 0], [0, 100], [100, 100]]),
    )
    fm_bag = np.array(
        [[0.0, 0.0], [10.0, 10.0], [0.0, 0.1], [10.0, 10.1]],
        dtype=np.float32,
    )
    # Colour groups are [0, 1] and [2, 3], unlike the FM groups [0, 2] and [1, 3].
    colour_descriptors = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    representation = YottixelRGB(
        params={"n_clusters": 2, "perc_selected": 50.0},
        config=SimpleNamespace(experiment=SimpleNamespace(random_state=0)),
    ).run(
        bag=fm_bag,
        sample=_make_sample(artifact_path),
        combo_cfg=SimpleNamespace(tile_px=256, tile_mpp=0.5),
        colour_descriptors=colour_descriptors,
        coords=np.array([[0, 0], [100, 0], [0, 100], [100, 100]], dtype=np.int32),
        tiling_id=bag_id,
    )

    np.testing.assert_array_equal(
        np.sort(representation.additional_data["selected_indices"]), np.array([0, 2])
    )
    np.testing.assert_array_equal(
        _sort_rows(representation.data), _sort_rows(fm_bag[[0, 2]])
    )


@pytest.mark.parametrize(
    ("colour_descriptor", "descriptor_shape"),
    [("mean_rgb", (2, 3)), ("histogram_rgb", (2, 768))],
)
def test_yottixel_rgb_load_sample_loads_fm_and_selected_colour_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    colour_descriptor: str,
    descriptor_shape: tuple[int, int],
) -> None:
    bag_id = "256px_0.5mpp"
    artifact_path = _write_coords_artifact(
        tmp_path,
        artifact_name="slide.h5",
        bag_id=bag_id,
        coords_xy=np.array([[0, 0], [1, 1]], dtype=np.int32),
    )
    fm_bag = np.arange(8, dtype=np.float32).reshape(2, 4)
    descriptor = np.ones(descriptor_shape, dtype=np.float32)
    calls: list[str] = []
    monkeypatch.setattr(
        "pathforge.slide_retrieval.representation_strategies.strategies.yottixel.features_io.read_features",
        lambda *args, **kwargs: fm_bag,
    )
    monkeypatch.setattr(
        "pathforge.slide_retrieval.representation_strategies.strategies.yottixel.resolve_sample_patch_mean_rgb",
        lambda **kwargs: calls.append("mean_rgb") or descriptor,
    )
    monkeypatch.setattr(
        "pathforge.slide_retrieval.representation_strategies.strategies.yottixel.resolve_sample_patch_histogram_rgb",
        lambda **kwargs: calls.append("histogram_rgb") or descriptor,
    )
    strategy = YottixelRGB(params={"colour_descriptor": colour_descriptor})

    payload = strategy.load_sample(
        index=0,
        sample=_make_sample(artifact_path),
        base_dataset=SimpleNamespace(tiling_id=bag_id, extractor_name="uni2"),
    )

    assert calls == [colour_descriptor]
    np.testing.assert_array_equal(payload["bag"], fm_bag)
    np.testing.assert_array_equal(payload["colour_descriptors"], descriptor)
    assert payload["coords"].shape == (2, 2)


def test_yottixel_rgb_descriptor_choice_changes_representation_identity() -> None:
    mean_id = build_retrieval_representation_id(
        "uni2", "yottixel-rgb", YottixelRGB().hyperparam_values()
    )
    histogram_id = build_retrieval_representation_id(
        "uni2",
        "yottixel-rgb",
        YottixelRGB({"colour_descriptor": "histogram_rgb"}).hyperparam_values(),
    )

    assert mean_id != histogram_id
