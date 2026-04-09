# tests/unit/test_feature_extraction_policy_report.py

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import pathbench.policy.feature_extraction as fe_mod
from pathbench.core.datasets.wsi_dataset import WSI
from pathbench.core.experiments.combinations import ComboConfig
from pathbench.policy.feature_extraction import FeatureExtractionPolicy


class _FakeFileHandleH5:
    def __init__(self, path: Path, mode: str = "a") -> None:
        self.path = path
        self.mode = mode
        self.h5: dict[str, Any] = {}

    def __enter__(self) -> "_FakeFileHandleH5":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeSlideProcessor:
    def __init__(self) -> None:
        self.load_calls = 0
        self.close_calls = 0
        self.get_thumbnail_calls = 0
        self.extract_features_calls = 0
        self.extract_patches_calls = 0
        self.segment_tissue_calls = 0

        self.thumbnail = np.zeros((40, 80, 3), dtype=np.uint8)

    def load_wsi(self, wsi: WSI) -> None:
        self.load_calls += 1

    def close_wsi(self, wsi: WSI) -> None:
        self.close_calls += 1

    def get_base_mpp(self, wsi: WSI) -> float:
        return 0.5

    def get_thumbnail(self, wsi: WSI, level: int = -1):
        self.get_thumbnail_calls += 1
        return self.thumbnail.copy(), 10.0, 10.0

    def extract_features(self, wsi: WSI, coords: np.ndarray, tiling_spec: dict, config: dict[str, Any]) -> np.ndarray:
        self.extract_features_calls += 1
        n = int(coords.shape[0])
        return np.zeros((n, 8), dtype=np.float32)

    def extract_patches(self, wsi: WSI, tissues, config: dict[str, Any] | None = None):
        self.extract_patches_calls += 1
        coords = np.array(
            [
                [0, 0, 256, 256, 0],
                [256, 0, 256, 256, 0],
            ],
            dtype=np.int32,
        )
        tiling_spec = {
            "tile_px": 256,
            "tile_mpp": 0.5,
            "stride_px": 256,
            "coord_space": "level0",
            "backend": "lazyslide",
        }
        return coords, tiling_spec

    def segment_tissue(self, wsi: WSI, config: dict[str, Any] | None = None):
        self.segment_tissue_calls += 1
        return []


def _make_policy(report: bool) -> FeatureExtractionPolicy:
    policy = FeatureExtractionPolicy.__new__(FeatureExtractionPolicy)
    policy.config = SimpleNamespace(
        experiment=SimpleNamespace(report=report, thumbnail=False)
    )
    return policy


def _make_wsi(tmp_path: Path) -> WSI:
    return WSI(
        slide="S1",
        patient="P1",
        category="cat",
        path=tmp_path / "S1.svs",
        artifact_path=tmp_path / "S1.h5",
    )


def _run_configs() -> dict[str, Any]:
    return {
        "seg_config": {"method": "otsu", "params": {}},
        "tile_config": {"tile_px": 256, "tile_mpp": 0.5, "params": {}},
        "feat_config": {"model": "dummy_extractor", "params": {}},
    }


def _default_coords() -> np.ndarray:
    return np.array(
        [
            [0, 0, 256, 256, 0],
            [256, 0, 256, 256, 0],
        ],
        dtype=np.int32,
    )


def _default_tiling_spec() -> dict[str, Any]:
    return {
        "tile_px": 256,
        "tile_mpp": 0.5,
        "stride_px": 256,
        "coord_space": "level0",
        "backend": "lazyslide",
    }


def _install_common_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    features_exist_sequence: list[bool],
    coords_are_valid: bool,
    overview_exists_sequence: list[bool] | None = None,
    coords: np.ndarray | None = None,
    tiling_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Patch H5 IO + renderer for policy tests and return a mutable state dict
    with call counters.
    """
    state: dict[str, Any] = {
        "features_exist_calls": [],
        "tiles_overview_exists_calls": 0,
        "write_tiles_overview_calls": 0,
        "thumbnail_image_exists_calls": 0,
        "thumbnail_spec_exists_calls": 0,
        "write_thumbnail_image_calls": 0,
        "write_thumbnail_spec_calls": 0,
        "write_coords_calls": 0,
        "write_tiling_spec_calls": 0,
        "write_features_calls": 0,
        "render_calls": 0,
        "render_thumbnail_calls": 0,
        "last_written_overview": None,
        "last_written_thumbnail": None,
        "last_written_thumbnail_spec": None,
    }

    coords_arr = _default_coords() if coords is None else coords
    tiling_spec_obj = _default_tiling_spec() if tiling_spec is None else tiling_spec
    overview_seq = list(overview_exists_sequence or [])

    monkeypatch.setattr(fe_mod, "FileHandleH5", _FakeFileHandleH5)

    def _no_logger_exception(*args, **kwargs):
        raise AssertionError("Unexpected exception inside FeatureExtractionPolicy._execute_wsi")

    monkeypatch.setattr(fe_mod.logger, "exception", _no_logger_exception)

    monkeypatch.setattr(
        fe_mod,
        "render_tiles_overview_image",
        lambda **kwargs: _render_stub(state, **kwargs),
    )
    monkeypatch.setattr(
        fe_mod,
        "render_thumbnail_image",
        lambda **kwargs: _render_thumbnail_stub(state, **kwargs),
    )

    monkeypatch.setattr(fe_mod.tiles_io, "coords_num_rows", lambda *a, **k: int(coords_arr.shape[0]))

    monkeypatch.setattr(fe_mod.tiles_io, "coords_exist", lambda *a, **k: bool(coords_are_valid))
    monkeypatch.setattr(fe_mod.tiles_io, "tiling_spec_matches", lambda *a, **k: bool(coords_are_valid))
    monkeypatch.setattr(fe_mod.tiles_io, "read_coords", lambda *a, **k: coords_arr.copy())
    monkeypatch.setattr(fe_mod.tiles_io, "read_tiling_spec", lambda *a, **k: dict(tiling_spec_obj))

    def _tiles_overview_exists(*args, **kwargs) -> bool:
        state["tiles_overview_exists_calls"] += 1
        if overview_seq:
            return bool(overview_seq.pop(0))
        return False

    monkeypatch.setattr(fe_mod.tiles_io, "tiles_overview_exists", _tiles_overview_exists)

    monkeypatch.setattr(
        fe_mod.thumbnail_io,
        "thumbnail_image_exists",
        lambda *a, **k: _bump_return_false(state, "thumbnail_image_exists_calls"),
    )
    monkeypatch.setattr(
        fe_mod.thumbnail_io,
        "thumbnail_spec_exists",
        lambda *a, **k: _bump_return_false(state, "thumbnail_spec_exists_calls"),
    )

    def _write_thumbnail_image(*args, **kwargs) -> None:
        state["write_thumbnail_image_calls"] += 1
        if len(args) >= 2:
            state["last_written_thumbnail"] = args[1]
        else:
            state["last_written_thumbnail"] = kwargs.get("image_bytes")

    def _write_thumbnail_spec(*args, **kwargs) -> None:
        state["write_thumbnail_spec_calls"] += 1
        if len(args) >= 2:
            state["last_written_thumbnail_spec"] = args[1]
        else:
            state["last_written_thumbnail_spec"] = kwargs.get("thumbnail_spec")

    monkeypatch.setattr(fe_mod.thumbnail_io, "write_thumbnail_image", _write_thumbnail_image)
    monkeypatch.setattr(fe_mod.thumbnail_io, "write_thumbnail_spec", _write_thumbnail_spec)

    def _write_tiles_overview(*args, **kwargs) -> None:
        state["write_tiles_overview_calls"] += 1
        if len(args) >= 3:
            state["last_written_overview"] = args[2]
        else:
            state["last_written_overview"] = kwargs.get("image_bytes")

    monkeypatch.setattr(fe_mod.tiles_io, "write_tiles_overview", _write_tiles_overview)
    monkeypatch.setattr(fe_mod.tiles_io, "write_coords", lambda *a, **k: _bump(state, "write_coords_calls"))
    monkeypatch.setattr(fe_mod.tiles_io, "write_tiling_spec", lambda *a, **k: _bump(state, "write_tiling_spec_calls"))

    def _features_exist(*args, **kwargs) -> bool:
        state["features_exist_calls"].append(kwargs.get("expected_rows"))
        if not features_exist_sequence:
            raise AssertionError("features_exist called more often than expected in test")
        return bool(features_exist_sequence.pop(0))

    monkeypatch.setattr(fe_mod.features_io, "features_exist", _features_exist)
    monkeypatch.setattr(fe_mod.features_io, "write_features", lambda *a, **k: _bump(state, "write_features_calls"))

    return state


def _render_stub(state: dict[str, Any], **kwargs) -> bytes:
    state["render_calls"] += 1
    assert "thumbnail_image" in kwargs
    assert "coords_array" in kwargs
    assert "downscale_x" in kwargs
    assert "downscale_y" in kwargs
    assert "base_mpp" in kwargs
    return b"\xff\xd8mockjpg"


def _render_thumbnail_stub(state: dict[str, Any], **kwargs) -> bytes:
    state["render_thumbnail_calls"] += 1
    assert "thumbnail_image" in kwargs
    return b"\xff\xd8thumbjpg"


def _bump(state: dict[str, Any], key: str) -> None:
    state[key] += 1


def _bump_return_false(state: dict[str, Any], key: str) -> bool:
    state[key] += 1
    return False


def _dataset_stub() -> Any:
    return SimpleNamespace(tissue_annotations_dir=None, name="ds")


def _execute(policy: FeatureExtractionPolicy, processor: _FakeSlideProcessor, tmp_path: Path) -> None:
    policy._execute_wsi(
        dataset=_dataset_stub(),
        wsi=_make_wsi(tmp_path),
        combo_cfg=ComboConfig(
            feature_extraction="dummy_extractor",
            tile_px=256,
            tile_mpp=0.5,
        ),
        slide_processor=processor,
        run_configs=_run_configs(),
    )


def test_report_false_no_overview_generation_attempt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    policy = _make_policy(report=False)
    processor = _FakeSlideProcessor()

    state = _install_common_mocks(
        monkeypatch,
        features_exist_sequence=[False, False],
        coords_are_valid=True,
        overview_exists_sequence=[],
    )

    _execute(policy, processor, tmp_path)

    assert processor.get_thumbnail_calls == 0
    assert state["render_calls"] == 0
    assert state["write_tiles_overview_calls"] == 0
    assert state["write_features_calls"] == 1


def test_thumbnail_false_no_thumbnail_generation_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy = _make_policy(report=False)
    processor = _FakeSlideProcessor()

    state = _install_common_mocks(
        monkeypatch,
        features_exist_sequence=[False, False],
        coords_are_valid=True,
        overview_exists_sequence=[],
    )

    _execute(policy, processor, tmp_path)

    assert state["render_thumbnail_calls"] == 0
    assert state["write_thumbnail_image_calls"] == 0
    assert state["write_thumbnail_spec_calls"] == 0


def test_report_true_tiles_reused_overview_missing_generates_and_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy = _make_policy(report=True)
    processor = _FakeSlideProcessor()

    state = _install_common_mocks(
        monkeypatch,
        features_exist_sequence=[False, False],
        coords_are_valid=True,
        overview_exists_sequence=[False],
    )

    _execute(policy, processor, tmp_path)

    assert processor.get_thumbnail_calls == 1
    assert state["render_calls"] == 1
    assert state["write_tiles_overview_calls"] == 1
    assert state["last_written_overview"] == b"\xff\xd8mockjpg"
    assert state["write_coords_calls"] == 0
    assert state["write_tiling_spec_calls"] == 0


def test_thumbnail_true_features_exist_but_thumbnail_missing_still_generates_thumbnail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy = _make_policy(report=False)
    policy.config.experiment.thumbnail = True
    processor = _FakeSlideProcessor()

    state = _install_common_mocks(
        monkeypatch,
        features_exist_sequence=[True, True],
        coords_are_valid=True,
        overview_exists_sequence=[],
    )

    _execute(policy, processor, tmp_path)

    assert processor.get_thumbnail_calls == 1
    assert state["render_thumbnail_calls"] == 1
    assert state["write_thumbnail_image_calls"] == 1
    assert state["write_thumbnail_spec_calls"] == 1
    assert processor.extract_features_calls == 0
    assert state["write_features_calls"] == 0


def test_report_true_tiles_reused_overview_exists_skips_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy = _make_policy(report=True)
    processor = _FakeSlideProcessor()

    state = _install_common_mocks(
        monkeypatch,
        features_exist_sequence=[False, False],
        coords_are_valid=True,
        overview_exists_sequence=[True],
    )

    _execute(policy, processor, tmp_path)

    assert processor.get_thumbnail_calls == 0
    assert state["render_calls"] == 0
    assert state["write_tiles_overview_calls"] == 0
    assert state["write_features_calls"] == 1


def test_report_true_features_exist_but_overview_missing_still_generates_overview(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy = _make_policy(report=True)
    processor = _FakeSlideProcessor()

    state = _install_common_mocks(
        monkeypatch,
        features_exist_sequence=[True, True],
        coords_are_valid=True,
        overview_exists_sequence=[False],
    )

    _execute(policy, processor, tmp_path)

    assert processor.get_thumbnail_calls == 1
    assert state["render_calls"] == 1
    assert state["write_tiles_overview_calls"] == 1
    assert processor.extract_features_calls == 0
    assert state["write_features_calls"] == 0


def test_report_true_tiles_newly_created_always_writes_overview_even_if_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy = _make_policy(report=True)
    processor = _FakeSlideProcessor()

    state = _install_common_mocks(
        monkeypatch,
        features_exist_sequence=[False, False],
        coords_are_valid=False,
        overview_exists_sequence=[True],
    )

    _execute(policy, processor, tmp_path)

    assert processor.extract_patches_calls == 1
    assert state["write_coords_calls"] == 1
    assert state["write_tiling_spec_calls"] == 1
    assert processor.get_thumbnail_calls == 1
    assert state["render_calls"] == 1
    assert state["write_tiles_overview_calls"] == 1


def test_thumbnail_true_tiles_newly_created_always_writes_thumbnail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy = _make_policy(report=False)
    policy.config.experiment.thumbnail = True
    processor = _FakeSlideProcessor()

    state = _install_common_mocks(
        monkeypatch,
        features_exist_sequence=[False, False],
        coords_are_valid=False,
        overview_exists_sequence=[],
    )

    _execute(policy, processor, tmp_path)

    assert processor.extract_patches_calls == 1
    assert processor.get_thumbnail_calls == 1
    assert state["render_thumbnail_calls"] == 1
    assert state["write_thumbnail_image_calls"] == 1
    assert state["write_thumbnail_spec_calls"] == 1
