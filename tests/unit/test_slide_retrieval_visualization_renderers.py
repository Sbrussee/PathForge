from __future__ import annotations

import numpy as np
from PIL import Image

from pathforge.slide_retrieval.visualization.service import (
    SlideRetrievalVisualizationService,
)
from pathforge.slide_retrieval.visualization.renderers import (
    render_retrieval_representation_image,
    render_retrieval_results_image,
)


def test_render_retrieval_results_image_builds_query_and_hit_grid() -> None:
    query_thumbnail = Image.new("RGB", (220, 180), (220, 220, 220))
    hit_panels = [
        (Image.new("RGB", (180, 200), (255, 200, 200)), [f"slide: hit_{idx}"])
        for idx in range(7)
    ]

    rendered = render_retrieval_results_image(
        query_thumbnail=query_thumbnail,
        query_lines=["slide: query_1", "category: tumor"],
        hit_panels=hit_panels,
    )

    assert rendered.mode == "RGB"
    assert rendered.width == 1740
    assert rendered.height > 700


def test_render_retrieval_results_image_wraps_long_metadata_lines() -> None:
    query_thumbnail = Image.new("RGB", (220, 180), (220, 220, 220))

    rendered = render_retrieval_results_image(
        query_thumbnail=query_thumbnail,
        query_lines=[
            "slide: very_long_slide_identifier_that_should_wrap_cleanly_across_lines_without_spilling",
            "category: tumor",
        ],
        hit_panels=[],
    )

    assert rendered.mode == "RGB"
    assert rendered.width == 1740
    assert rendered.height > 380


def test_render_retrieval_representation_image_supports_missing_patch_strip() -> None:
    thumbnail = Image.new("RGB", (400, 300), (245, 245, 245))
    coords_array = np.array(
        [
            [0, 0, 64, 64, 0],
            [128, 64, 64, 64, 0],
            [256, 128, 64, 64, 0],
        ],
        dtype=np.int32,
    )
    group_ids = np.array([0, 1, 2], dtype=np.int32)
    selected_coords = np.array([[0, 0], [256, 128]], dtype=np.int32)

    rendered = render_retrieval_representation_image(
        thumbnail_image=thumbnail,
        downscale_x=4.0,
        downscale_y=4.0,
        coords_array=coords_array,
        tiling_spec=None,
        base_mpp=None,
        group_ids=group_ids,
        selected_coords=selected_coords,
        tissue_polygons=None,
        patch_strip_images=[],
        patch_group_ids=[],
    )

    assert rendered.mode == "RGB"
    assert rendered.width > 1340
    assert rendered.height == 680


def test_render_retrieval_representation_image_supports_tissue_crop_legend_and_full_width_patch_grid() -> None:
    thumbnail = Image.new("RGB", (400, 300), (255, 255, 255))
    for x in range(120, 281):
        for y in range(90, 211):
            thumbnail.putpixel((x, y), (210, 160, 160))

    coords_array = np.array(
        [
            [140, 100, 64, 64, 0],
            [200, 100, 64, 64, 0],
            [140, 160, 64, 64, 0],
            [200, 160, 64, 64, 0],
        ],
        dtype=np.int32,
    )
    group_ids = np.array([0, 1, 0, 1], dtype=np.int32)
    selected_coords = np.array(
        [
            [140, 100],
            [200, 100],
            [140, 160],
            [200, 160],
            [140, 100],
            [200, 100],
            [140, 160],
            [200, 160],
        ],
        dtype=np.int32,
    )
    tissue_polygons = [
        [
            [
                [120.0, 90.0],
                [280.0, 90.0],
                [280.0, 210.0],
                [120.0, 210.0],
                [120.0, 90.0],
            ],
        ]
    ]
    patch_images = [
        Image.new("RGB", (96, 96), (200, 220 - idx * 10, 200))
        for idx in range(8)
    ]

    rendered = render_retrieval_representation_image(
        thumbnail_image=thumbnail,
        downscale_x=1.0,
        downscale_y=1.0,
        coords_array=coords_array,
        tiling_spec={"tile_px": 64, "tile_mpp": 1.0},
        base_mpp=1.0,
        group_ids=group_ids,
        selected_coords=selected_coords,
        tissue_polygons=tissue_polygons,
        patch_strip_images=patch_images,
        patch_group_ids=[0, 1, 0, 1, 0, 1, 0, 1],
    )

    assert rendered.mode == "RGB"
    assert rendered.width > 1340
    assert rendered.height == 864


def test_visualization_service_uses_flat_visualization_output_dirs(tmp_path) -> None:
    service = SlideRetrievalVisualizationService.__new__(
        SlideRetrievalVisualizationService
    )
    service.run_dir = tmp_path / "run_001"
    service.visualization_root = None

    output_dir = service._ensure_output_dir("retrieval_results")

    assert output_dir == tmp_path / "run_001" / "vis_retrieval_results"
    assert output_dir.is_dir()


def test_visualization_service_uses_flat_representation_visualization_root(
    tmp_path,
) -> None:
    service = SlideRetrievalVisualizationService.__new__(
        SlideRetrievalVisualizationService
    )
    service.run_dir = tmp_path / "run_001"
    service.visualization_root = tmp_path / "representation_root"

    output_dir = service._ensure_output_dir("retrieval_representation")

    assert output_dir == tmp_path / "representation_root" / "vis_retrieval_representation"
    assert output_dir.is_dir()
