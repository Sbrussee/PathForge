from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image
from PIL import ImageOps

from pathbench.core.visualization.tiles_overview import _to_pil_rgb


def render_thumbnail_image(
    *,
    thumbnail_image: Any,
    jpeg_quality: int = 65,
    max_long_side: int | None = 1200,
) -> bytes:
    """
    Render one full-slide thumbnail image as JPEG bytes.

    Inputs:
    - `thumbnail_image`: PIL image or numpy array thumbnail.
    - `jpeg_quality`: JPEG quality used for encoded bytes.
    - `max_long_side`: optional resize cap applied before encoding.

    Returns:
    - `bytes`: encoded JPEG image.
    """
    pil_img = _to_pil_rgb(thumbnail_image)

    if max_long_side is not None and int(max_long_side) > 0:
        img_w0, img_h0 = pil_img.size
        longest = max(img_w0, img_h0)
        if longest > int(max_long_side):
            resize_scale = float(max_long_side) / float(longest)
            new_w = max(1, int(round(img_w0 * resize_scale)))
            new_h = max(1, int(round(img_h0 * resize_scale)))
            try:
                resample = Image.Resampling.BILINEAR
            except AttributeError:  # pragma: no cover - Pillow compatibility
                resample = Image.BILINEAR  # type: ignore[attr-defined]
            pil_img = pil_img.resize((new_w, new_h), resample=resample)

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
    except TypeError:  # pragma: no cover - Pillow compatibility
        save_kwargs.pop("subsampling", None)
        save_kwargs.pop("progressive", None)
        pil_img.save(buf, **save_kwargs)
    return buf.getvalue()


def decode_thumbnail_image(image_bytes: bytes) -> Image.Image:
    """Decode one stored thumbnail byte payload into a PIL RGB image."""
    image = Image.open(BytesIO(image_bytes))
    image.load()
    return image.convert("RGB")


def fit_image_to_canvas(
    image: Image.Image,
    *,
    canvas_size: tuple[int, int],
    background_rgb: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Contain one image inside a fixed-size canvas while preserving aspect ratio."""
    image = image.convert("RGB")
    fitted = ImageOps.contain(image, canvas_size)
    canvas = Image.new("RGB", canvas_size, background_rgb)
    offset_x = (canvas_size[0] - fitted.width) // 2
    offset_y = (canvas_size[1] - fitted.height) // 2
    canvas.paste(fitted, (offset_x, offset_y))
    return canvas


def crop_thumbnail_background(
    image: Image.Image,
    *,
    threshold: int = 245,
    padding: int = 6,
) -> Image.Image:
    """
    Crop near-white background from one thumbnail image.

    Returns the input image unchanged when no foreground bbox can be detected.
    """
    rgb_image = image.convert("RGB")
    bbox = rgb_image.point(
        lambda value: 0 if value >= threshold else 255
    ).convert("L").getbbox()
    if bbox is None:
        return rgb_image

    left = max(0, bbox[0] - padding)
    top = max(0, bbox[1] - padding)
    right = min(rgb_image.width, bbox[2] + padding)
    bottom = min(rgb_image.height, bbox[3] + padding)
    return rgb_image.crop((left, top, right, bottom))


def project_level0_to_thumbnail(
    *,
    x_level0: float,
    y_level0: float,
    downscale_x: float,
    downscale_y: float,
) -> tuple[float, float]:
    """
    Project one level-0 point into thumbnail pixel space.

    Returns:
    - `(x_thumb, y_thumb)`: thumbnail-space coordinates.
    """
    downscale_x = float(downscale_x)
    downscale_y = float(downscale_y)
    if downscale_x <= 0 or downscale_y <= 0:
        raise ValueError("downscale_x and downscale_y must be > 0.")
    return (float(x_level0) / downscale_x, float(y_level0) / downscale_y)
