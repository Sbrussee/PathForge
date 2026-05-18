from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pathbench.slide_retrieval.representation_strategies.mean_rgb as mean_rgb_mod
from pathbench.core.io.h5 import descriptors as descriptors_io
from pathbench.core.io.h5 import tiles as tiles_io
from pathbench.core.io.h5.base import FileHandleH5


def _write_artifact(
    artifact_path: Path,
    *,
    bag_id: str,
    coords: np.ndarray,
    tiling_spec: dict[str, object] | None = None,
    mean_rgb: np.ndarray | None = None,
) -> None:
    with FileHandleH5(artifact_path, mode="a") as slide_artifact:
        tiles_io.write_coords(slide_artifact, bag_id, coords)
        tiles_io.write_tiling_spec(
            slide_artifact,
            bag_id,
            tiling_spec
            or {
                "tile_px": 256,
                "tile_mpp": 0.5,
                "stride_px": 256,
                "coord_space": "level0",
                "backend": "lazyslide",
            },
        )
    if mean_rgb is not None:
        retrieval_artifact_path = mean_rgb_mod._slide_retrieval_artifact_path(
            slide_artifact_path=artifact_path,
            slide_id="slide-1",
        )
        with FileHandleH5(retrieval_artifact_path, mode="a") as retrieval_artifact:
            descriptors_io.write_descriptor(
                retrieval_artifact,
                bag_id,
                "mean_rgb",
                mean_rgb,
            )


def _make_sample(artifact_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        slide_ids=["slide-1"],
        artifact_paths=[artifact_path],
        metadata={"dataset": "dataset-a"},
    )


def _make_config(slides_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        slide_processing=SimpleNamespace(backend="test_backend"),
        datasets=[
            SimpleNamespace(
                name="dataset-a",
                slides_dir=str(slides_dir),
            )
        ],
    )


def test_resolve_sample_patch_mean_rgb_reads_existing_descriptors_without_recomputing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bag_id = "256px_0.5mpp"
    artifact_path = tmp_path / "slide-1.h5"
    coords = np.array([[0, 0, 256, 256, 0]], dtype=np.int32)
    expected = np.array([[0.2, 0.4, 0.6]], dtype=np.float32)
    _write_artifact(artifact_path, bag_id=bag_id, coords=coords, mean_rgb=expected)

    class _UnusedProcessor:
        def load_wsi(self, wsi):
            raise AssertionError("Processor should not be used when descriptors exist.")

        def close_wsi(self, wsi):
            raise AssertionError("Processor should not be used when descriptors exist.")

    monkeypatch.setattr(mean_rgb_mod, "_build_slide_processor", lambda **_: _UnusedProcessor())

    resolved = mean_rgb_mod.resolve_sample_patch_mean_rgb(
        sample=_make_sample(artifact_path),
        bag_id=bag_id,
        config=_make_config(tmp_path),
    )

    np.testing.assert_allclose(resolved, expected)


def test_resolve_sample_patch_mean_rgb_computes_and_persists_missing_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bag_id = "256px_0.5mpp"
    artifact_path = tmp_path / "slide-1.h5"
    coords = np.array(
        [[0, 0, 256, 256, 0], [10, 10, 256, 256, 0]],
        dtype=np.int32,
    )
    _write_artifact(artifact_path, bag_id=bag_id, coords=coords)

    slide_path = tmp_path / "slide-1.svs"
    slide_path.write_bytes(b"fake-slide")

    class _Processor:
        def __init__(self) -> None:
            self.loaded_paths: list[Path] = []

        def load_wsi(self, wsi) -> None:
            self.loaded_paths.append(Path(wsi.path))

        def close_wsi(self, wsi) -> None:
            return None

        def read_patch_region(self, wsi, x, y, width, height, level) -> np.ndarray:
            _ = wsi
            _ = (x, y, width, height, level)
            return np.full((2, 2, 3), 128, dtype=np.uint8)

    processor = _Processor()
    monkeypatch.setattr(mean_rgb_mod, "_build_slide_processor", lambda **_: processor)

    resolved = mean_rgb_mod.resolve_sample_patch_mean_rgb(
        sample=_make_sample(artifact_path),
        bag_id=bag_id,
        config=_make_config(tmp_path),
    )

    assert processor.loaded_paths == [slide_path]
    np.testing.assert_allclose(
        resolved,
        np.full((2, 3), 128.0 / 255.0, dtype=np.float32),
    )

    retrieval_artifact_path = mean_rgb_mod._slide_retrieval_artifact_path(
        slide_artifact_path=artifact_path,
        slide_id="slide-1",
    )
    with FileHandleH5(retrieval_artifact_path, mode="r") as retrieval_artifact:
        stored = descriptors_io.read_descriptor(retrieval_artifact, bag_id, "mean_rgb")
    np.testing.assert_allclose(stored, resolved)


def test_resolve_sample_patch_mean_rgb_raises_when_slide_file_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bag_id = "256px_0.5mpp"
    artifact_path = tmp_path / "slide-1.h5"
    coords = np.array([[0, 0, 256, 256, 0]], dtype=np.int32)
    _write_artifact(artifact_path, bag_id=bag_id, coords=coords)

    class _Processor:
        def load_wsi(self, wsi) -> None:
            raise AssertionError("Slide load should not be reached without a source slide.")

        def close_wsi(self, wsi) -> None:
            return None

    monkeypatch.setattr(mean_rgb_mod, "_build_slide_processor", lambda **_: _Processor())

    with pytest.raises(FileNotFoundError, match="Missing stored patch mean RGB descriptors"):
        mean_rgb_mod.resolve_sample_patch_mean_rgb(
            sample=_make_sample(artifact_path),
            bag_id=bag_id,
            config=_make_config(tmp_path),
        )
