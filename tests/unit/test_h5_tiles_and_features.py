# tests/unit/test_h5_tiles_and_features.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathbench.core.io.h5.base import FileHandleH5, write_array_dataset
from pathbench.core.io.h5 import tiles as tiles_io
from pathbench.core.io.h5 import features as features_io


def test_tiles_coords_and_tiling_spec_roundtrip(tmp_path: Path) -> None:
    h5_path = tmp_path / "S1.h5"
    bag_id = "256px_0.5mpp"

    coords_in = np.array(
        [
            [0, 0, 256, 256, 0],
            [256, 0, 256, 256, 0],
            [0, 256, 256, 256, 0],
        ],
        dtype=np.int32,
    )
    spec_in = {"tile_px": 256, "tile_mpp": 0.5}

    with FileHandleH5(h5_path, mode="a") as f:
        assert tiles_io.coords_exist(f, bag_id) is False
        tiles_io.write_coords(f, bag_id, coords_in)
        assert tiles_io.coords_exist(f, bag_id) is True

        coords_out = tiles_io.read_coords(f, bag_id)
        assert coords_out.dtype == np.int32
        np.testing.assert_array_equal(coords_out, coords_in)

        assert tiles_io.tiling_spec_exists(f, bag_id) is False
        tiles_io.write_tiling_spec(f, bag_id, spec_in)
        assert tiles_io.tiling_spec_exists(f, bag_id) is True

        spec_out = tiles_io.read_tiling_spec(f, bag_id)
        assert spec_out == spec_in

        assert tiles_io.tiling_spec_matches(f, bag_id, spec_in) is True
        assert tiles_io.tiling_spec_matches(f, bag_id, {"tile_px": 128, "tile_mpp": 0.5}) is False


def test_write_coords_rejects_wrong_shape(tmp_path: Path) -> None:
    h5_path = tmp_path / "S1.h5"
    bag_id = "256px_0.5mpp"

    bad = np.array([[1, 2, 3]], dtype=np.int32)  # (1,3) not (N,5)

    with FileHandleH5(h5_path, mode="a") as f:
        with pytest.raises(ValueError):
            tiles_io.write_coords(f, bag_id, bad)


def test_features_roundtrip_and_validation(tmp_path: Path) -> None:
    h5_path = tmp_path / "S1.h5"
    bag_id = "256px_0.5mpp"
    extractor = "dummy_extractor"

    feats_in = np.random.rand(3, 5).astype(np.float32)

    with FileHandleH5(h5_path, mode="a") as f:
        assert features_io.features_exist(f, bag_id, extractor) is False
        features_io.write_features(f, bag_id, extractor, feats_in)
        assert features_io.features_exist(f, bag_id, extractor) is True

        feats_out = features_io.read_features(f, bag_id, extractor)
        assert feats_out.dtype == np.float32
        assert feats_out.shape == (3, 5)
        np.testing.assert_allclose(feats_out, feats_in)

        with pytest.raises(ValueError):
            features_io.write_features(f, bag_id, extractor, np.array([1, 2, 3], dtype=np.float32))


def test_read_features_rejects_non_2d(tmp_path: Path) -> None:
    h5_path = tmp_path / "S1.h5"
    bag_id = "256px_0.5mpp"
    extractor = "bad_extractor"

    with FileHandleH5(h5_path, mode="a") as f:
        # Bypass write_features and write a 1D array at the features dataset path
        layout_path = f"bags/{bag_id}/features/{extractor}"
        write_array_dataset(f.h5, layout_path, np.array([1, 2, 3], dtype=np.float32), dtype=np.float32)

        with pytest.raises(ValueError) as excinfo:
            _ = features_io.read_features(f, bag_id, extractor)

        assert "features must have shape (N,D)" in str(excinfo.value)


def test_tiles_overview_roundtrip_and_overwrite(tmp_path: Path) -> None:
    h5_path = tmp_path / "S1.h5"
    bag_id = "256px_0.5mpp"

    overview_v1 = b"\xff\xd8\xff\xe0fakejpeg_v1"
    overview_v2 = b"\xff\xd8\xff\xe0fakejpeg_v2_longer"

    with FileHandleH5(h5_path, mode="a") as f:
        assert tiles_io.tiles_overview_exists(f, bag_id) is False

        tiles_io.write_tiles_overview(f, bag_id, overview_v1)
        assert tiles_io.tiles_overview_exists(f, bag_id) is True

        out_v1 = tiles_io.read_tiles_overview(f, bag_id)
        assert isinstance(out_v1, (bytes, bytearray))
        assert bytes(out_v1) == overview_v1

        # overwrite should replace existing bytes
        tiles_io.write_tiles_overview(f, bag_id, overview_v2)
        out_v2 = tiles_io.read_tiles_overview(f, bag_id)
        assert isinstance(out_v2, (bytes, bytearray))
        assert bytes(out_v2) == overview_v2


def test_write_tiles_overview_rejects_non_bytes(tmp_path: Path) -> None:
    h5_path = tmp_path / "S1.h5"
    bag_id = "256px_0.5mpp"

    with FileHandleH5(h5_path, mode="a") as f:
        with pytest.raises((TypeError, ValueError)):
            tiles_io.write_tiles_overview(f, bag_id, "not-bytes")  # type: ignore[arg-type]


def test_write_array_dataset_allows_scalar_utf8_values(tmp_path: Path) -> None:
    h5_path = tmp_path / "S1.h5"
    dataset_path = "bags/256px_0.5mpp/retrieval_representations/repr_1/additional_data/dataset_name"

    with FileHandleH5(h5_path, mode="a") as f:
        write_array_dataset(
            f.h5,
            dataset_path,
            np.asarray("dataset-a"),
            dtype=np.asarray("dataset-a").dtype,
        )

        dset = f.h5[dataset_path]
        assert tuple(dset.shape) == ()
        raw = dset[()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        assert str(raw) == "dataset-a"
