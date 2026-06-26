from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pathforge.core.datasets.wsi_dataset import WSI
from pathforge.core.slide_processing.lazyslide import LazySlideProcessor


class FakeLoadedWSI(dict):
    def __init__(self, mpp):
        super().__init__()
        self.properties = SimpleNamespace(mpp=mpp)
        self.attrs = {}


def test_get_base_mpp_uses_loaded_slide_mpp() -> None:
    proc = LazySlideProcessor()

    wsi = WSI(
        slide="slide_1",
        patient="patient_1",
        category="cat",
        path=None,
        artifact_path=None,
        fallback_mpp=0.5,
    )
    wsi._obj = FakeLoadedWSI(mpp=0.25)

    result = proc.get_base_mpp(wsi)

    assert result == 0.25


def test_get_base_mpp_uses_fallback_mpp_when_loaded_mpp_missing() -> None:
    proc = LazySlideProcessor()

    wsi = WSI(
        slide="slide_1",
        patient="patient_1",
        category="cat",
        path=None,
        artifact_path=None,
        fallback_mpp=0.5,
    )
    wsi._obj = FakeLoadedWSI(mpp=None)

    result = proc.get_base_mpp(wsi)

    assert result == 0.5


def test_get_base_mpp_raises_when_loaded_mpp_and_fallback_missing() -> None:
    proc = LazySlideProcessor()

    wsi = WSI(
        slide="slide_1",
        patient="patient_1",
        category="cat",
        path=None,
        artifact_path=None,
        fallback_mpp=None,
    )
    wsi._obj = FakeLoadedWSI(mpp=None)

    with pytest.raises(RuntimeError, match="base MPP is missing"):
        _ = proc.get_base_mpp(wsi)


def test_extract_patches_passes_slide_mpp_from_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = LazySlideProcessor()

    wsi = WSI(
        slide="slide_1",
        patient="patient_1",
        category="cat",
        path=None,
        artifact_path=None,
        fallback_mpp=0.5,
    )
    wsi._obj = FakeLoadedWSI(mpp=None)

    captured: dict[str, object] = {}

    def fake_policy_tissues_to_backend(tissues):
        return "fake_tissues"

    def fake_tile_tissues(*, wsi, **params):
        captured["params"] = params
        wsi["tiles"] = "fake_tiles"
        wsi.attrs["tile_spec"] = {"tiles": {"width": 256, "mpp": 1.0, "ops_level": 0, "ops_downsample": 2.0}}

    def fake_backend_tiles_to_policy_coords(tiles_table, tile_spec_obj):
        return np.array([[0, 0, 512, 512, 0]], dtype=np.int32)

    def fake_backend_tile_spec_to_policy_tiling_spec(*, config, tile_spec_obj):
        return {
            "tile_px": 256,
            "tile_mpp": 1.0,
            "stride_px": 256,
            "coord_space": "level0",
            "backend": "lazyslide",
        }

    monkeypatch.setattr(proc, "_policy_tissues_to_backend", fake_policy_tissues_to_backend)
    monkeypatch.setattr(proc, "_backend_tiles_to_policy_coords", fake_backend_tiles_to_policy_coords)
    monkeypatch.setattr(proc, "_backend_tile_spec_to_policy_tiling_spec", fake_backend_tile_spec_to_policy_tiling_spec)
    monkeypatch.setattr("pathforge.core.slide_processing.lazyslide.zs.pp.tile_tissues", fake_tile_tissues)

    coords, tiling_spec = proc.extract_patches(
        wsi,
        tissues=[[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]],
        config={"tile_px": 256, "tile_mpp": 1.0, "params": {}},
    )

    assert captured["params"]["tile_px"] == 256
    assert captured["params"]["mpp"] == 1.0
    assert captured["params"]["slide_mpp"] == 0.5

    assert coords.shape == (1, 5)
    assert tiling_spec["tile_px"] == 256
    assert tiling_spec["tile_mpp"] == 1.0