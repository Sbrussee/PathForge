from __future__ import annotations

import math
from typing import Iterable, Sequence

from PIL import Image, ImageColor, ImageDraw, ImageFont
import numpy as np

from pathforge.core.visualization.thumbnail import (
    crop_thumbnail_to_tissue_bounds,
    crop_thumbnail_background,
    fit_image_to_canvas,
    project_level0_to_thumbnail,
)


RESULT_THUMB_SIZE = (340, 280)
REPRESENTATION_THUMB_SIZE = (640, 640)
PATCH_THUMB_SIZE = (144, 144)
GRID_COLS = 5
GRID_ROWS = 2
REPRESENTATION_PATCH_COLS = 8
PADDING = 20
TEXT_LINE_HEIGHT = 22
LEGEND_SWATCH_SIZE = 16
LEGEND_LINE_HEIGHT = 20
LEGEND_PADDING = 10
LEGEND_FONT_SIZE = 13
TISSUE_CROP_BORDER_PX = 12
METADATA_FONT_SIZE = 16

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
    query_panel = _build_metadata_panel(
        image=query_thumbnail,
        text_lines=query_lines,
        panel_size=RESULT_THUMB_SIZE,
    )

    hit_cells = [
        _build_metadata_panel(
            image=thumbnail,
            text_lines=text_lines,
            panel_size=RESULT_THUMB_SIZE,
        )
        for thumbnail, text_lines in hit_panels[: GRID_COLS * GRID_ROWS]
    ]

    row_heights: list[int] = []
    normalized_rows: list[list[Image.Image]] = []
    if hit_cells:
        for row_idx in range(math.ceil(len(hit_cells) / GRID_COLS)):
            row_cells = hit_cells[row_idx * GRID_COLS : (row_idx + 1) * GRID_COLS]
            row_height = max(cell.height for cell in row_cells)
            row_heights.append(row_height)
            normalized_rows.append(
                [_pad_panel_to_height(cell, target_height=row_height) for cell in row_cells]
            )

    total_width = PADDING * 2 + GRID_COLS * RESULT_THUMB_SIZE[0]
    total_height = PADDING * 3 + query_panel.height + sum(row_heights)
    if row_heights:
        total_height += PADDING * len(row_heights)

    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    query_x = (total_width - query_panel.width) // 2
    query_y = PADDING
    canvas.paste(query_panel, (query_x, query_y))

    row_y = query_y + query_panel.height + PADDING
    for row_idx, row_cells in enumerate(normalized_rows):
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
    tiling_spec: dict[str, object] | None,
    base_mpp: float | None,
    group_ids: np.ndarray | None,
    selected_coords: np.ndarray | None,
    tissue_polygons: Sequence[Sequence[Sequence[Sequence[float]]]] | None,
    patch_strip_images: Iterable[Image.Image],
    patch_group_ids: Sequence[int | None] | None,
) -> Image.Image:
    """Render one side-by-side retrieval representation summary image."""
    base_thumbnail = thumbnail_image.convert("RGB")
    grouped_thumbnail = base_thumbnail.copy()
    selected_thumbnail = base_thumbnail.copy()
    has_group_overlay = (
        group_ids is not None and coords_array.shape[0] == int(group_ids.shape[0])
    )

    patch_width, patch_height = _resolve_patch_span(
        coords_array=coords_array,
        tiling_spec=tiling_spec,
        base_mpp=base_mpp,
        downscale_x=downscale_x,
        downscale_y=downscale_y,
    )

    if has_group_overlay:
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

    grouped_thumbnail = crop_thumbnail_to_tissue_bounds(
        grouped_thumbnail,
        tissue_polygons=tissue_polygons,
        downscale_x=downscale_x,
        downscale_y=downscale_y,
        border_px=TISSUE_CROP_BORDER_PX,
    )
    selected_thumbnail = crop_thumbnail_to_tissue_bounds(
        selected_thumbnail,
        tissue_polygons=tissue_polygons,
        downscale_x=downscale_x,
        downscale_y=downscale_y,
        border_px=TISSUE_CROP_BORDER_PX,
    )

    legend_panel = None
    legend_panel_width = 0
    if has_group_overlay:
        legend_panel = _build_group_legend_panel(
            group_ids=np.asarray(group_ids),
            canvas_height=REPRESENTATION_THUMB_SIZE[1],
        )
        legend_panel_width = legend_panel.width

    left_panel = fit_image_to_canvas(
        grouped_thumbnail,
        canvas_size=REPRESENTATION_THUMB_SIZE,
    )
    right_panel = fit_image_to_canvas(
        selected_thumbnail,
        canvas_size=REPRESENTATION_THUMB_SIZE,
    )
    total_width = PADDING * 4 + 2 * REPRESENTATION_THUMB_SIZE[0] + legend_panel_width
    panel_height = PADDING * 2 + REPRESENTATION_THUMB_SIZE[1]

    patch_images = list(patch_strip_images)
    patch_rows = math.ceil(len(patch_images) / REPRESENTATION_PATCH_COLS) if patch_images else 0
    patch_strip_height = 0
    if patch_rows > 0:
        patch_strip_height = PADDING + patch_rows * PATCH_THUMB_SIZE[1] + PADDING

    total_height = panel_height + patch_strip_height
    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    left_x = PADDING
    canvas.paste(left_panel, (left_x, PADDING))

    legend_x = left_x + REPRESENTATION_THUMB_SIZE[0] + PADDING
    if legend_panel is not None:
        canvas.paste(legend_panel, (legend_x, PADDING))

    right_x = legend_x + legend_panel_width + PADDING
    canvas.paste(right_panel, (right_x, PADDING))

    if patch_images:
        row_y = panel_height + PADDING
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        grid_width = REPRESENTATION_PATCH_COLS * PATCH_THUMB_SIZE[0]
        grid_x0 = max(PADDING, (total_width - grid_width) // 2)
        for idx, patch_image in enumerate(patch_images):
            row_idx, col_idx = divmod(idx, REPRESENTATION_PATCH_COLS)
            patch_canvas = fit_image_to_canvas(
                patch_image,
                canvas_size=PATCH_THUMB_SIZE,
            )
            x = grid_x0 + col_idx * PATCH_THUMB_SIZE[0]
            y = row_y + row_idx * PATCH_THUMB_SIZE[1]
            canvas.paste(patch_canvas, (x, y))
            label = _format_patch_label(
                idx,
                None
                if patch_group_ids is None or idx >= len(patch_group_ids)
                else patch_group_ids[idx],
            )
            label_width, label_height = _measure_text(label, font=font)
            draw.rectangle(
                [(x + 2, y + 2), (x + 10 + label_width, y + 8 + label_height)],
                fill=(255, 255, 255),
                outline=(255, 255, 255),
                width=1,
            )
            draw.text((x + 5, y + 4), label, fill=(0, 0, 0), font=font)

    return canvas


def _build_metadata_panel(
    *,
    image: Image.Image,
    text_lines: list[str],
    panel_size: tuple[int, int],
) -> Image.Image:
    image_canvas = fit_image_to_canvas(
        crop_thumbnail_background(image),
        canvas_size=panel_size,
    )
    key_font = _load_font(size=METADATA_FONT_SIZE, bold=True)
    value_font = _load_font(size=METADATA_FONT_SIZE, bold=False)
    wrapped_lines: list[tuple[str, list[str]]] = []
    text_width = panel_size[0] - 16
    for line in text_lines:
        key, value = _split_metadata_line(str(line))
        key_width, _ = _measure_text(key, font=key_font)
        first_line_width = max(40, text_width - key_width)
        continuation_width = max(40, text_width)
        value_lines = _wrap_text_to_width(
            value,
            font=value_font,
            first_line_width=first_line_width,
            continuation_width=continuation_width,
        )
        wrapped_lines.append((key, value_lines))

    total_text_rows = sum(max(1, len(value_lines)) for _, value_lines in wrapped_lines)
    text_height = max(TEXT_LINE_HEIGHT * max(1, total_text_rows) + PADDING + 10, 92)
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
    for key, value_lines in wrapped_lines:
        draw.text((8, text_y), key, fill=(0, 0, 0), font=key_font)
        key_width, _ = _measure_text(key, font=key_font)
        first_value = value_lines[0] if value_lines else "-"
        draw.text((8 + key_width, text_y), first_value, fill=(0, 0, 0), font=value_font)
        text_y += TEXT_LINE_HEIGHT
        for continuation in value_lines[1:]:
            draw.text((8, text_y), continuation, fill=(0, 0, 0), font=value_font)
            text_y += TEXT_LINE_HEIGHT
    return panel


def _pad_panel_to_height(
    panel: Image.Image,
    *,
    target_height: int,
) -> Image.Image:
    if panel.height >= target_height:
        return panel

    padded = Image.new("RGB", (panel.width, target_height), (255, 255, 255))
    padded.paste(panel, (0, 0))
    draw = ImageDraw.Draw(padded)
    draw.line(
        [(0, panel.height - 1), (panel.width - 1, panel.height - 1)],
        fill=(255, 255, 255),
        width=1,
    )
    draw.rectangle(
        [(0, 0), (padded.width - 1, padded.height - 1)],
        outline=(0, 0, 0),
        width=1,
    )
    return padded


def _resolve_patch_span(
    *,
    coords_array: np.ndarray,
    tiling_spec: dict[str, object] | None,
    base_mpp: float | None,
    downscale_x: float,
    downscale_y: float,
) -> tuple[int, int]:
    if tiling_spec is not None and base_mpp is not None:
        try:
            tile_px = int(tiling_spec["tile_px"])
            tile_mpp = float(tiling_spec["tile_mpp"])
            base_mpp = float(base_mpp)
            if tile_px > 0 and tile_mpp > 0 and base_mpp > 0:
                tile_size_um = float(tile_px) * float(tile_mpp)
                tile_w_l0 = max(1.0, tile_size_um / base_mpp)
                tile_h_l0 = max(1.0, tile_size_um / base_mpp)
                return (
                    max(2, int(round(tile_w_l0 / float(downscale_x)))),
                    max(2, int(round(tile_h_l0 / float(downscale_y)))),
                )
        except (KeyError, TypeError, ValueError):
            pass

    if coords_array.size == 0:
        return (16, 16)
    return (
        max(2, int(round(float(np.median(coords_array[:, 2])) / float(downscale_x)))),
        max(2, int(round(float(np.median(coords_array[:, 3])) / float(downscale_y)))),
    )


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
        label = str(idx)
        text_width, text_height = _measure_text(label, font=font)
        text_x = left + max(0, (marker_size - text_width) // 2)
        text_y = top + max(0, (marker_size - text_height) // 2)
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)


def _build_group_legend_panel(
    *,
    group_ids: np.ndarray,
    canvas_height: int,
) -> Image.Image:
    unique_group_ids = sorted({int(group_id) for group_id in np.asarray(group_ids).tolist()})
    if not unique_group_ids:
        return Image.new("RGB", (0, canvas_height), (255, 255, 255))

    font = _load_font(size=LEGEND_FONT_SIZE, bold=False)
    labels = [f"group {group_id}" for group_id in unique_group_ids]
    text_widths = [_measure_text(label, font=font)[0] for label in labels]
    items_per_column = max(
        1,
        (canvas_height - 2 * LEGEND_PADDING) // LEGEND_LINE_HEIGHT,
    )
    num_columns = max(1, int(math.ceil(len(labels) / items_per_column)))
    column_width = (
        LEGEND_PADDING * 3
        + LEGEND_SWATCH_SIZE
        + max(text_widths, default=0)
    )
    legend_width = num_columns * column_width + LEGEND_PADDING
    panel = Image.new("RGB", (legend_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(panel)

    for idx, group_id in enumerate(unique_group_ids):
        col_idx, row_idx = divmod(idx, items_per_column)
        color_rgb = ImageColor.getrgb(_PALETTE[group_id % len(_PALETTE)])
        col_x = LEGEND_PADDING + col_idx * column_width
        row_y = LEGEND_PADDING + row_idx * LEGEND_LINE_HEIGHT
        draw.rectangle(
            [
                (col_x, row_y + 2),
                (
                    col_x + LEGEND_SWATCH_SIZE,
                    row_y + 2 + LEGEND_SWATCH_SIZE,
                ),
            ],
            fill=color_rgb,
            outline=(0, 0, 0),
            width=1,
        )
        draw.text(
            (col_x + LEGEND_PADDING + LEGEND_SWATCH_SIZE, row_y + 1),
            f"group {group_id}",
            fill=(0, 0, 0),
            font=font,
        )
    return panel


def _format_patch_label(idx: int, group_id: int | None) -> str:
    if group_id is None:
        return f"#{idx}"
    return f"#{idx} g{int(group_id)}"


def _split_metadata_line(line: str) -> tuple[str, str]:
    if ":" not in line:
        return (line.strip() + ": ", "-")
    key, value = line.split(":", 1)
    return (key.strip() + ": ", value.strip() or "-")


def _measure_text(text: str, *, font: ImageFont.ImageFont) -> tuple[int, int]:
    try:
        left, top, right, bottom = font.getbbox(text)
        return (int(right - left), int(bottom - top))
    except AttributeError:
        return tuple(int(v) for v in font.getsize(text))




def _wrap_text_to_width(
    text: str,
    *,
    font: ImageFont.ImageFont,
    first_line_width: int,
    continuation_width: int,
) -> list[str]:
    stripped = str(text).strip()
    if not stripped:
        return ["-"]

    words = stripped.split()
    if not words:
        return [stripped]

    lines: list[str] = []
    current = ""
    current_limit = first_line_width

    for word in words:
        candidate = word if not current else f"{current} {word}"
        candidate_width, _ = _measure_text(candidate, font=font)
        if candidate_width <= current_limit:
            current = candidate
            continue

        if current:
            lines.append(current)
            current = word
            current_limit = continuation_width
            continue

        split_chunks = _split_long_token(word, font=font, max_width=current_limit)
        lines.extend(split_chunks[:-1])
        current = split_chunks[-1]
        current_limit = continuation_width

    if current:
        lines.append(current)
    return lines or ["-"]


def _split_long_token(
    token: str,
    *,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    if not token:
        return [""]

    chunks: list[str] = []
    current = ""
    for character in token:
        candidate = current + character
        candidate_width, _ = _measure_text(candidate, font=font)
        if candidate_width <= max_width or not current:
            current = candidate
            continue
        chunks.append(current)
        current = character

    if current:
        chunks.append(current)
    return chunks or [token]


def _load_font(*, size: int, bold: bool) -> ImageFont.ImageFont:
    font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(font_name, size=size)
    except OSError:
        return ImageFont.load_default()
