from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

import pathforge.inference.heatmaps as heatmaps_mod
from pathforge.core.io.h5.base import FileHandleH5
from pathforge.core.io.h5 import heatmaps as heatmap_io
from pathforge.core.io.h5 import tiles as tiles_io
from pathforge.inference.heatmaps import (
    _coords_to_pixel_rectangles,
    _top_tiles_grid_shape,
    create_inference_heatmap,
)
from pathforge.utils.registries import EXPLAINERS


@dataclass(frozen=True)
class _FakeHeatmap:
    coords: torch.Tensor
    scores: torch.Tensor


class _FakeHeatmapExplainer:
    def initialize(self, config):
        self.config = config

    def explain(self, input):
        coords = input["coords"]
        scores = input["instance_scores"].float()
        if "mask" in input:
            mask = input["mask"].bool()
            coords = coords[mask]
            scores = scores[mask]
        scores = (scores - scores.min()) / torch.clamp(
            scores.max() - scores.min(), min=1e-12
        )
        return _FakeHeatmap(coords=coords, scores=scores)


def _register_fake_explainer_once(name: str) -> None:
    if EXPLAINERS.is_available(name):
        return
    EXPLAINERS.register(name)(_FakeHeatmapExplainer)


def test_create_inference_heatmap_persists_h5_and_json_sidecar(tmp_path, monkeypatch):
    _register_fake_explainer_once("fake_inference_heatmap")
    artifact_path = tmp_path / "slide.h5"
    slide_path = tmp_path / "slide.svs"
    scores_path = tmp_path / "scores.npy"
    output_path = tmp_path / "heatmap.json"
    image_output_path = tmp_path / "heatmap.png"
    slide_path.write_bytes(b"fake slide placeholder")
    np.save(scores_path, np.asarray([0.2, 0.5, 1.0], dtype=np.float32))

    captured_slide_paths: list[object] = []

    def _fake_read_top_tile_images(**kwargs):
        captured_slide_paths.append(kwargs["slide_path"])
        return [
            np.full((32, 32, 3), fill_value=50 * (idx + 1), dtype=np.uint8)
            for idx in range(kwargs["scores"].shape[0])
        ]

    monkeypatch.setattr(
        heatmaps_mod, "_read_top_tile_images", _fake_read_top_tile_images
    )

    with FileHandleH5(artifact_path, mode="a") as slide_artifact:
        tiles_io.write_coords(
            slide_artifact,
            "256px_0.5mpp",
            np.asarray(
                [
                    [0, 0, 256, 256, 0],
                    [256, 0, 256, 256, 0],
                    [512, 0, 256, 256, 0],
                ],
                dtype=np.int32,
            ),
        )
        tiles_io.write_tiling_spec(
            slide_artifact,
            "256px_0.5mpp",
            {
                "tile_px": 256,
                "tile_mpp": 0.5,
                "coord_space": "level0",
                "tiles_overview_downscale_x": 2.0,
                "tiles_overview_downscale_y": 2.0,
            },
        )
        tiles_io.write_tiles_overview(
            slide_artifact,
            "256px_0.5mpp",
            _overview_bytes(width=384, height=192),
        )

    result = create_inference_heatmap(
        artifact_path=artifact_path,
        bag_id="256px_0.5mpp",
        scores_path=scores_path,
        heatmap_backend="fake_inference_heatmap",
        heatmap_name="attention",
        output_path=output_path,
        image_output_path=image_output_path,
        model_path="model.ckpt",
        slide_path=slide_path,
    )

    assert result.num_points == 3
    assert output_path.exists()
    assert image_output_path.exists()
    assert result.smoothed_image_output_path is not None
    assert result.smoothed_image_output_path.exists()
    assert result.top_tiles_output_path is not None
    assert result.top_tiles_output_path.exists()
    assert result.image_output_path == image_output_path
    assert captured_slide_paths == [slide_path]
    with FileHandleH5(artifact_path, mode="r") as slide_artifact:
        heatmap = heatmap_io.read_prediction_heatmap(
            slide_artifact, "256px_0.5mpp", "attention"
        )

    assert heatmap["coords"].shape == (3, 2)
    assert heatmap["scores"].shape == (3,)
    assert heatmap["metadata"]["backend"] == "fake_inference_heatmap"


def test_coords_to_pixel_rectangles_preserves_tile_square_alignment() -> None:
    rectangles = _coords_to_pixel_rectangles(
        coords=np.asarray([[0.0, 0.0], [200.0, 0.0]], dtype=np.float32),
        tile_sizes=np.asarray([[200.0, 200.0], [200.0, 200.0]], dtype=np.float32),
        downscale_x=2.0,
        downscale_y=2.0,
    )

    assert rectangles.tolist() == [[0, 0, 100, 100], [100, 0, 100, 100]]


def test_top_tiles_grid_shape_prefers_compact_matrix() -> None:
    assert _top_tiles_grid_shape(1) == (1, 1)
    assert _top_tiles_grid_shape(4) == (2, 2)
    assert _top_tiles_grid_shape(5) == (2, 3)
    assert _top_tiles_grid_shape(10) == (3, 4)


def test_create_inference_heatmap_applies_mask(tmp_path):
    _register_fake_explainer_once("fake_inference_heatmap")
    artifact_path = tmp_path / "slide.h5"
    scores_path = tmp_path / "scores.npy"
    coords_path = tmp_path / "coords.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(scores_path, np.asarray([0.2, 0.5, 1.0], dtype=np.float32))
    np.save(coords_path, np.asarray([[0, 0], [256, 0], [512, 0]], dtype=np.float32))
    np.save(mask_path, np.asarray([1, 0, 1], dtype=np.uint8))

    result = create_inference_heatmap(
        artifact_path=artifact_path,
        bag_id="256px_0.5mpp",
        scores_path=scores_path,
        coords_path=coords_path,
        mask_path=mask_path,
        heatmap_backend="fake_inference_heatmap",
        heatmap_name="masked_attention",
    )

    assert result.num_points == 2
    with FileHandleH5(artifact_path, mode="r") as slide_artifact:
        heatmap = heatmap_io.read_prediction_heatmap(
            slide_artifact, "256px_0.5mpp", "masked_attention"
        )

    assert heatmap["coords"].tolist() == [[0.0, 0.0], [512.0, 0.0]]


def _overview_bytes(
    *,
    width: int,
    height: int,
    color: tuple[int, int, int] = (255, 255, 255),
) -> bytes:
    """Create deterministic JPEG overview bytes for rendering tests."""

    from io import BytesIO

    image = Image.new("RGB", (width, height), color=color)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()
