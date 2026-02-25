# src/pathbench/core/visualization/tiling.py
from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _to_pil_rgb(image: Any) -> Image.Image:
    """
    Convert a PIL image or numpy array to a PIL RGB image.

    Supported inputs:
    - numpy HxW (grayscale)
    - numpy HxWx3 (RGB)
    - numpy HxWx4 (RGBA)
    - PIL.Image.Image
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image

    arr = np.asarray(image)

    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")

    if arr.ndim == 3:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.shape[2] == 3:
            return Image.fromarray(arr, mode="RGB")

        if arr.shape[2] == 4:
            return Image.fromarray(arr, mode="RGBA").convert("RGB")

    raise ValueError(
        "Unsupported thumbnail_image shape/type for rendering: "
        f"type={type(image)!r}, shape={getattr(arr, 'shape', None)!r}"
    )


def render_tiles_overview_image(
    *,
    thumbnail_image: Any,
    coords_array: np.ndarray,
    downscale_x: float,
    downscale_y: float,
    slide_id: str,  # kept for call-site compatibility (unused)
    tiling_spec: dict[str, Any] | None = None,  # kept for call-site compatibility (unused)
    jpeg_quality: int = 65,
    max_long_side: int | None = 1200,
) -> bytes:
    """
    Render a tile overview image (thumbnail + tile grid overlay) and return JPEG bytes.

    Notes:
    - No title/text is drawn here (PDF draws text consistently).
    - `slide_id` and `tiling_spec` are kept in the signature so existing policy code
      does not need to change.
    - `max_long_side` reduces stored image size substantially for large thumbnails.
    """
    # ---- validate coords ----
    coords = np.asarray(coords_array)
    if coords.ndim != 2 or coords.shape[1] != 5:
        raise ValueError(f"coords_array must have shape (N,5), got {coords.shape}")

    # ---- validate scaling ----
    try:
        downscale_x = float(downscale_x)
        downscale_y = float(downscale_y)
    except Exception as e:
        raise ValueError("downscale_x and downscale_y must be numeric") from e

    if downscale_x <= 0 or downscale_y <= 0:
        raise ValueError(
            f"downscale_x and downscale_y must be > 0, got {downscale_x}, {downscale_y}"
        )

    # ---- normalize thumbnail to RGB PIL image ----
    pil_img = _to_pil_rgb(thumbnail_image)

    # ---- optional resize to reduce storage footprint ----
    if max_long_side is not None and int(max_long_side) > 0:
        img_w0, img_h0 = pil_img.size
        longest = max(img_w0, img_h0)
        if longest > int(max_long_side):
            resize_scale = float(max_long_side) / float(longest)
            new_w = max(1, int(round(img_w0 * resize_scale)))
            new_h = max(1, int(round(img_h0 * resize_scale)))

            # Pillow compatibility (older/newer versions)
            try:
                resample = Image.Resampling.BILINEAR  # Pillow >= 9
            except AttributeError:
                resample = Image.BILINEAR  # type: ignore[attr-defined]

            pil_img = pil_img.resize((new_w, new_h), resample=resample)

            # Thumbnail got smaller, so effective downscale increases
            downscale_x = downscale_x / resize_scale
            downscale_y = downscale_y / resize_scale

    draw = ImageDraw.Draw(pil_img)
    img_w, img_h = pil_img.size

    # ---- draw tile grid (black outlines) ----
    # coords columns: [x, y, read_w, read_h, read_level]
    # x/y are level-0 coordinates. We map them to thumbnail space using the
    # provided downscale factors.
    for row in coords:
        x0 = int(round(float(row[0]) / downscale_x))
        y0 = int(round(float(row[1]) / downscale_y))
        w0 = max(1, int(round(float(row[2]) / downscale_x)))
        h0 = max(1, int(round(float(row[3]) / downscale_y)))

        x1 = x0 + w0
        y1 = y0 + h0

        # Skip tiles completely outside thumbnail bounds
        if x1 < 0 or y1 < 0 or x0 >= img_w or y0 >= img_h:
            continue

        draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=1)

    # ---- encode JPEG (small storage footprint) ----
    buf = BytesIO()
    save_kwargs = {
        "format": "JPEG",
        "quality": int(jpeg_quality),
        "optimize": True,
        "subsampling": 2,   # 4:2:0 (smaller files)
        "progressive": False,
    }

    try:
        pil_img.save(buf, **save_kwargs)
    except TypeError:
        # Fallback for older Pillow versions that may not support some kwargs
        save_kwargs.pop("subsampling", None)
        save_kwargs.pop("progressive", None)
        pil_img.save(buf, **save_kwargs)

    return buf.getvalue()