from __future__ import annotations

import math
from typing import Iterable

from PIL import Image, ImageColor, ImageDraw, ImageFont
import numpy as np

from pathbench.core.visualization.thumbnail import (
    fit_image_to_canvas,
    project_level0_to_thumbnail,
)


RESULT_THUMB_SIZE = (280, 280)
REPRESENTATION_THUMB_SIZE = (640, 640)
PATCH_THUMB_SIZE = (128, 128)
GRID_COLS = 5
GRID_ROWS = 2
PADDING = 20
TEXT_LINE_HEIGHT = 18

_PALETTE = [
    "#D55E00",
    "#0072B2",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#999999",
]


def render_retrieval_results_image(
    *,
    query_thumbnail: Image.Image,
    query_lines: list[str],
    hit_panels: list[tuple[Image.Image, list[str]]],
) -> Image.Image:
    """Render one query + top-hit retrieval result summary image."""
    font = ImageFont.load_default()
    query_panel = _build_metadata_panel(
        image=query_thumbnail,
        text_lines=query_lines,
        panel_size=RESULT_THUMB_SIZE,
        font=font,
    )

    hit_cells = [
        _build_metadata_panel(
            image=thumbnail,
            text_lines=text_lines,
            panel_size=RESULT_THUMB_SIZE,
            font=font,
        )
        for thumbnail, text_lines in hit_panels[: GRID_COLS * GRID_ROWS]
    ]

    row_heights: list[int] = []
    if hit_cells:
        for row_idx in range(math.ceil(len(hit_cells) / GRID_COLS)):
            row_cells = hit_cells[row_idx * GRID_COLS : (row_idx + 1) * GRID_COLS]
            row_heights.append(max(cell.height for cell in row_cells))

    total_width = PADDING * 2 + GRID_COLS * RESULT_THUMB_SIZE[0]
    total_height = PADDING * 3 + query_panel.height + sum(row_heights)
    if row_heights:
        total_height += PADDING * len(row_heights)

    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    query_x = (total_width - query_panel.width) // 2
    query_y = PADDING
    canvas.paste(query_panel, (query_x, query_y))

    row_y = query_y + query_panel.height + PADDING
    for row_idx in range(math.ceil(len(hit_cells) / GRID_COLS)):
        row_cells = hit_cells[row_idx * GRID_COLS : (row_idx + 1) * GRID_COLS]
        for col_idx, cell in enumerate(row_cells):
            x = PADDING + col_idx * RESULT_THUMB_SIZE[0]
            canvas.paste(cell, (x, row_y))
        row_y += row_heights[row_idx] + PADDING

    return canvas


def render_retrieval_representation_image(
    *,
    thumbnail_image: Image.Image,
    downscale_x: float,
    downscale_y: float,
    coords_array: np.ndarray,
    group_ids: np.ndarray | None,
    selected_coords: np.ndarray | None,
    patch_strip_images: Iterable[Image.Image],
) -> Image.Image:
    """Render one side-by-side retrieval representation summary image."""
    base_thumbnail = thumbnail_image.convert("RGB")
    grouped_thumbnail = base_thumbnail.copy()
    selected_thumbnail = base_thumbnail.copy()

    patch_width = _resolve_patch_span(coords_array[:, 2] if coords_array.size else None, downscale_x)
    patch_height = _resolve_patch_span(coords_array[:, 3] if coords_array.size else None, downscale_y)

    if group_ids is not None and coords_array.shape[0] == int(group_ids.shape[0]):
        _draw_group_overlay(
            grouped_thumbnail,
            coords_array=coords_array,
            group_ids=group_ids,
            downscale_x=downscale_x,
            downscale_y=downscale_y,
            patch_width=patch_width,
            patch_height=patch_height,
        )

    if selected_coords is not None:
        _draw_selected_overlay(
            selected_thumbnail,
            selected_coords=selected_coords,
            downscale_x=downscale_x,
            downscale_y=downscale_y,
            patch_width=patch_width,
            patch_height=patch_height,
        )

    panels = [
        fit_image_to_canvas(grouped_thumbnail, canvas_size=REPRESENTATION_THUMB_SIZE),
        fit_image_to_canvas(selected_thumbnail, canvas_size=REPRESENTATION_THUMB_SIZE),
    ]
    total_width = PADDING * 3 + 2 * REPRESENTATION_THUMB_SIZE[0]
    panel_height = PADDING * 2 + REPRESENTATION_THUMB_SIZE[1]

    patch_images = list(patch_strip_images)
    patch_rows = math.ceil(len(patch_images) / GRID_COLS) if patch_images else 0
    patch_strip_height = 0
    if patch_rows > 0:
        patch_strip_height = PADDING + patch_rows * PATCH_THUMB_SIZE[1] + PADDING

    total_height = panel_height + patch_strip_height
    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    for panel_idx, panel in enumerate(panels):
        x = PADDING + panel_idx * (REPRESENTATION_THUMB_SIZE[0] + PADDING)
        canvas.paste(panel, (x, PADDING))

    if patch_images:
        row_y = panel_height + PADDING
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        for idx, patch_image in enumerate(patch_images):
            row_idx, col_idx = divmod(idx, GRID_COLS)
            patch_canvas = fit_image_to_canvas(
                patch_image,
                canvas_size=PATCH_THUMB_SIZE,
            )
            x = PADDING + col_idx * PATCH_THUMB_SIZE[0]
            y = row_y + row_idx * PATCH_THUMB_SIZE[1]
            canvas.paste(patch_canvas, (x, y))
            draw.text((x + 4, y + 4), str(idx), fill=(0, 0, 0), font=font)

    return canvas


def _build_metadata_panel(
    *,
    image: Image.Image,
    text_lines: list[str],
    panel_size: tuple[int, int],
    font: ImageFont.ImageFont,
) -> Image.Image:
    image_canvas = fit_image_to_canvas(
        crop_thumbnail_background(image),
        canvas_size=panel_size,
    )
    text_height = max(TEXT_LINE_HEIGHT * max(1, len(text_lines)) + PADDING, 80)
    panel = Image.new(
        "RGB",
        (panel_size[0], panel_size[1] + text_height),
        (255, 255, 255),
    )
    panel.paste(image_canvas, (0, 0))
    draw = ImageDraw.Draw(panel)
    draw.rectangle(
        [(0, 0), (panel.width - 1, panel.height - 1)],
        outline=(0, 0, 0),
        width=1,
    )
    text_y = panel_size[1] + 8
    for line in text_lines:
        draw.text((8, text_y), str(line), fill=(0, 0, 0), font=font)
        text_y += TEXT_LINE_HEIGHT
    return panel


def _resolve_patch_span(values: np.ndarray | None, downscale: float) -> int:
    if values is None or values.size == 0:
        return 16
    return max(2, int(round(float(np.median(values)) / float(downscale))))


def _draw_group_overlay(
    image: Image.Image,
    *,
    coords_array: np.ndarray,
    group_ids: np.ndarray,
    downscale_x: float,
    downscale_y: float,
    patch_width: int,
    patch_height: int,
) -> None:
    rgba_image = image.convert("RGBA")
    overlay = Image.new("RGBA", rgba_image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for idx, coord_row in enumerate(coords_array):
        x_thumb, y_thumb = project_level0_to_thumbnail(
            x_level0=float(coord_row[0]),
            y_level0=float(coord_row[1]),
            downscale_x=downscale_x,
            downscale_y=downscale_y,
        )
        color_rgb = ImageColor.getrgb(_PALETTE[int(group_ids[idx]) % len(_PALETTE)])
        draw.rectangle(
            [
                int(round(x_thumb)),
                int(round(y_thumb)),
                int(round(x_thumb)) + patch_width,
                int(round(y_thumb)) + patch_height,
            ],
            fill=(*color_rgb, 96),
            outline=(0, 0, 0, 180),
            width=1,
        )
    composited = Image.alpha_composite(rgba_image, overlay).convert("RGB")
    image.paste(composited)


def _draw_selected_overlay(
    image: Image.Image,
    *,
    selected_coords: np.ndarray,
    downscale_x: float,
    downscale_y: float,
    patch_width: int,
    patch_height: int,
) -> None:
    draw = ImageDraw.Draw(image)
    marker_size = max(12, min(28, patch_width, patch_height))
    font = ImageFont.load_default()
    for idx, coord in enumerate(np.asarray(selected_coords)):
        x_thumb, y_thumb = project_level0_to_thumbnail(
            x_level0=float(coord[0]),
            y_level0=float(coord[1]),
            downscale_x=downscale_x,
            downscale_y=downscale_y,
        )
        left = int(round(x_thumb))
        top = int(round(y_thumb))
        draw.rectangle(
            [(left, top), (left + marker_size, top + marker_size)],
            fill=(0, 0, 0),
            outline=(0, 0, 0),
            width=1,
        )
        draw.text((left + 3, top + 2), str(idx), fill=(255, 255, 255), font=font)
