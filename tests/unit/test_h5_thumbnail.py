from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pathforge.config.config import Config
from pathforge.core.io.slide_artifacts import thumbnail as thumbnail_io
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.core.visualization.thumbnail import (
    project_level0_to_thumbnail,
    render_thumbnail_image,
)


def test_thumbnail_image_and_spec_roundtrip(tmp_path: Path) -> None:
    h5_path = tmp_path / "S1.h5"
    image_bytes = b"\xff\xd8\xff\xe0thumbjpeg"
    spec_in = {
        "image_format": "jpeg",
        "coord_space": "level0",
        "thumbnail_level": -1,
        "downscale_x": 10.0,
        "downscale_y": 20.0,
    }

    with FileHandleH5(h5_path, mode="a") as slide_artifact:
        assert thumbnail_io.thumbnail_image_exists(slide_artifact) is False
        assert thumbnail_io.thumbnail_spec_exists(slide_artifact) is False

        thumbnail_io.write_thumbnail_image(slide_artifact, image_bytes)
        thumbnail_io.write_thumbnail_spec(slide_artifact, spec_in)

        assert thumbnail_io.thumbnail_image_exists(slide_artifact) is True
        assert thumbnail_io.thumbnail_spec_exists(slide_artifact) is True
        assert thumbnail_io.read_thumbnail_image(slide_artifact) == image_bytes
        assert thumbnail_io.read_thumbnail_spec(slide_artifact) == spec_in


def test_write_thumbnail_image_rejects_non_bytes(tmp_path: Path) -> None:
    h5_path = tmp_path / "S1.h5"
    with FileHandleH5(h5_path, mode="a") as slide_artifact:
        with pytest.raises((TypeError, ValueError)):
            thumbnail_io.write_thumbnail_image(
                slide_artifact,
                "not-bytes",  # type: ignore[arg-type]
            )


def test_write_thumbnail_spec_rejects_invalid_payload(tmp_path: Path) -> None:
    h5_path = tmp_path / "S1.h5"
    with FileHandleH5(h5_path, mode="a") as slide_artifact:
        with pytest.raises(ValueError):
            thumbnail_io.write_thumbnail_spec(
                slide_artifact,
                {
                    "image_format": "jpeg",
                    "coord_space": "level0",
                    "thumbnail_level": -1,
                    "downscale_x": 10.0,
                },
            )


def test_render_thumbnail_image_returns_non_empty_jpeg() -> None:
    thumbnail = np.full((50, 100, 3), 180, dtype=np.uint8)
    image_bytes = render_thumbnail_image(thumbnail_image=thumbnail)
    assert image_bytes[:2] == b"\xff\xd8"
    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    assert image.mode == "RGB"
    assert image.size == (100, 50)


def test_project_level0_to_thumbnail_uses_stored_downscale() -> None:
    x_thumb, y_thumb = project_level0_to_thumbnail(
        x_level0=200.0,
        y_level0=400.0,
        downscale_x=10.0,
        downscale_y=20.0,
    )
    assert x_thumb == 20.0
    assert y_thumb == 20.0


def test_config_thumbnail_defaults_to_false(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  project_name: demo",
                "  annotation_file: dummy.csv",
                "  mode: feature_extraction",
                "slide_processing:",
                "  backend: lazyslide",
                "datasets: []",
                "benchmark_parameters:",
                "  tile_px: [256]",
                "  tile_mpp: [0.5]",
                "  feature_extraction: ['resnet18']",
                "  mil: []",
            ]
        ),
        encoding="utf-8",
    )

    cfg = Config.from_yaml(yaml_path)
    assert cfg.experiment.thumbnail is False

