from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class TilesOverviewRenderResult:
    """JPEG overview image plus the thumbnail-space mapping used to render it.

    Attributes:
        image_bytes: JPEG-encoded thumbnail with tile grid overlay.
        downscale_x: Effective level-0-to-thumbnail x downscale.
        downscale_y: Effective level-0-to-thumbnail y downscale.
        image_width_px: Rendered overview width in pixels.
        image_height_px: Rendered overview height in pixels.
    """

    image_bytes: bytes
    downscale_x: float
    downscale_y: float
    image_width_px: int
    image_height_px: int


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
    tiling_spec: dict[str, Any] | None = None,
    base_mpp: float | None = None,
    jpeg_quality: int = 65,
    max_long_side: int | None = 1200,
) -> bytes:
    """Backward-compatible wrapper returning only JPEG bytes."""

    return render_tiles_overview(
        thumbnail_image=thumbnail_image,
        coords_array=coords_array,
        downscale_x=downscale_x,
        downscale_y=downscale_y,
        slide_id=slide_id,
        tiling_spec=tiling_spec,
        base_mpp=base_mpp,
        jpeg_quality=jpeg_quality,
        max_long_side=max_long_side,
    ).image_bytes


def render_tiles_overview(
    *,
    thumbnail_image: Any,
    coords_array: np.ndarray,
    downscale_x: float,
    downscale_y: float,
    slide_id: str,
    tiling_spec: dict[str, Any] | None = None,
    base_mpp: float | None = None,
    jpeg_quality: int = 65,
    max_long_side: int | None = 1200,
) -> TilesOverviewRenderResult:
    """
    Render a tile overview image (thumbnail + tile grid overlay) and return JPEG bytes.

    The tile overlay size is computed exactly from:
    - tile_px (output tile width in pixels at target mpp)
    - tile_mpp (target microns-per-pixel)
    - base_mpp (slide level-0 microns-per-pixel)

    So:
        tile_size_um = tile_px * tile_mpp
        tile_size_level0_px = tile_size_um / base_mpp

    Notes:
    - `coords[:, 0:2]` are level-0 top-left coordinates.
    - `read_w/read_h/read_level` are not used for overlay sizing.
    - No title/text is drawn here (PDF draws text consistently).
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

    # ---- validate tiling spec + base mpp ----
    if tiling_spec is None:
        raise ValueError("tiling_spec is required to render the tile overlay.")
    if base_mpp is None:
        raise ValueError("base_mpp is required to render the tile overlay.")

    try:
        tile_px = int(tiling_spec["tile_px"])
        tile_mpp = float(tiling_spec["tile_mpp"])
    except KeyError as e:
        raise ValueError(
            "tiling_spec must contain 'tile_px' and 'tile_mpp' to render the tile overlay."
        ) from e

    if tile_px <= 0 or tile_mpp <= 0:
        raise ValueError(
            f"tiling_spec values must be > 0, got tile_px={tile_px}, tile_mpp={tile_mpp}"
        )

    try:
        base_mpp = float(base_mpp)
    except Exception as e:
        raise ValueError("base_mpp must be numeric") from e

    if base_mpp <= 0:
        raise ValueError(f"base_mpp must be > 0, got {base_mpp}")

    coord_space = tiling_spec.get("coord_space")
    if coord_space is not None and str(coord_space) != "level0":
        raise ValueError(
            f"Unsupported coord_space for tile overlay: {coord_space!r}. Expected 'level0'."
        )

    # ---- compute exact tile footprint in level-0 pixels ----
    tile_size_um = float(tile_px) * float(tile_mpp)
    tile_w_l0 = max(1.0, tile_size_um / float(base_mpp))
    tile_h_l0 = max(1.0, tile_size_um / float(base_mpp))

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

    # ---- precompute thumbnail-space tile size ----
    tile_w_thumb = max(1, int(round(tile_w_l0 / downscale_x)))
    tile_h_thumb = max(1, int(round(tile_h_l0 / downscale_y)))

    # ---- draw tile grid ----
    # coords columns: [x, y, read_w, read_h, read_level]
    # x/y are authoritative level-0 top-left coordinates.
    for row in coords:
        x_l0 = float(row[0])
        y_l0 = float(row[1])

        x0 = int(round(x_l0 / downscale_x))
        y0 = int(round(y_l0 / downscale_y))

        # Pillow rectangles are inclusive, so subtract 1 for exact size.
        x1 = x0 + tile_w_thumb - 1
        y1 = y0 + tile_h_thumb - 1

        # Skip tiles completely outside thumbnail bounds
        if x1 < 0 or y1 < 0 or x0 >= img_w or y0 >= img_h:
            continue

        # Clip partially visible tiles
        x0_clip = max(0, x0)
        y0_clip = max(0, y0)
        x1_clip = min(img_w - 1, x1)
        y1_clip = min(img_h - 1, y1)

        draw.rectangle([x0_clip, y0_clip, x1_clip, y1_clip], outline=(0, 0, 0), width=1)

    # ---- encode JPEG ----
    buf = BytesIO()
    save_kwargs = {
        "format": "JPEG",
        "quality": int(jpeg_quality),
        "optimize": True,
        "subsampling": 2,
        "progressive": False,
    }

    try:
        pil_img.save(buf, **save_kwargs)
    except TypeError:
        save_kwargs.pop("subsampling", None)
        save_kwargs.pop("progressive", None)
        pil_img.save(buf, **save_kwargs)

    return TilesOverviewRenderResult(
        image_bytes=buf.getvalue(),
        downscale_x=float(downscale_x),
        downscale_y=float(downscale_y),
        image_width_px=int(img_w),
        image_height_px=int(img_h),
    )
