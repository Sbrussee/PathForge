from __future__ import annotations

import numpy as np
from PIL import Image

from pathbench.slide_retrieval.visualization.renderers import (
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
    assert rendered.width == 1440
    assert rendered.height > 700


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
        group_ids=group_ids,
        selected_coords=selected_coords,
        patch_strip_images=[],
    )

    assert rendered.mode == "RGB"
    assert rendered.size == (1340, 680)
