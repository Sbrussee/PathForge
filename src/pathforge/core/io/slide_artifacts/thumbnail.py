from __future__ import annotations

from typing import Any

import numpy as np

from pathforge.core.io.slide_artifacts.base import (
    FileHandleH5,
    get_dataset,
    is_complete,
    read_array_dataset,
    read_json_dataset,
    write_array_dataset,
    write_json_dataset,
)
from pathforge.core.io.slide_artifacts.layout import DEFAULT_LAYOUT, H5Layout


def thumbnail_image_exists(
    slide_artifact: FileHandleH5,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> bool:
    """Return whether the slide-level thumbnail image dataset is present."""
    path = layout.thumbnail_image_dataset()
    dset = get_dataset(slide_artifact.h5, path)
    if dset is None or not is_complete(dset):
        return False
    shape = getattr(dset, "shape", None)
    return bool(shape and len(shape) == 1)


def write_thumbnail_image(
    slide_artifact: FileHandleH5,
    image_bytes: bytes,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
    """Store one encoded slide thumbnail as a 1D uint8 dataset."""
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        raise TypeError(f"image_bytes must be bytes-like. Got {type(image_bytes)}.")

    arr = np.frombuffer(memoryview(image_bytes), dtype=np.uint8)
    if arr.ndim != 1:
        raise ValueError("image_bytes must produce a 1D uint8 buffer.")

    write_array_dataset(
        slide_artifact.h5,
        layout.thumbnail_image_dataset(),
        arr,
        dtype=np.uint8,
    )


def read_thumbnail_image(
    slide_artifact: FileHandleH5,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> bytes:
    """Read the encoded slide thumbnail bytes."""
    path = layout.thumbnail_image_dataset()
    dset = get_dataset(slide_artifact.h5, path)
    if dset is None:
        raise KeyError(f"Missing thumbnail image dataset: {path}")
    if not is_complete(dset):
        raise ValueError(f"Thumbnail image dataset is incomplete: {path}")

    payload = read_array_dataset(slide_artifact.h5, path)
    if isinstance(payload, np.ndarray):
        arr = np.asarray(payload, dtype=np.uint8)
        if arr.ndim != 1:
            raise ValueError(
                f"thumbnail image must be a 1D uint8 array. Got shape {arr.shape}."
            )
        return arr.tobytes()

    try:
        return bytes(payload)
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise ValueError(
            "thumbnail image payload could not be converted to bytes "
            f"(type={type(payload)!r})."
        ) from exc


def thumbnail_spec_exists(
    slide_artifact: FileHandleH5,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> bool:
    """Return whether the slide-level thumbnail spec dataset is present."""
    path = layout.thumbnail_spec_dataset()
    try:
        dset = get_dataset(slide_artifact.h5, path)
        if dset is None or not is_complete(dset):
            return False
        _validate_thumbnail_spec_value(read_json_dataset(slide_artifact.h5, path))
    except Exception:
        return False
    return True


def write_thumbnail_spec(
    slide_artifact: FileHandleH5,
    thumbnail_spec: dict[str, Any],
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
    """Write one JSON thumbnail spec for slide-level thumbnail interpretation."""
    _validate_thumbnail_spec_value(thumbnail_spec)
    write_json_dataset(slide_artifact.h5, layout.thumbnail_spec_dataset(), thumbnail_spec)


def read_thumbnail_spec(
    slide_artifact: FileHandleH5,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> dict[str, Any]:
    """Read the JSON thumbnail spec."""
    path = layout.thumbnail_spec_dataset()
    dset = get_dataset(slide_artifact.h5, path)
    if dset is None:
        raise KeyError(f"Missing thumbnail spec dataset: {path}")
    if not is_complete(dset):
        raise ValueError(f"Thumbnail spec dataset is incomplete: {path}")

    spec = read_json_dataset(slide_artifact.h5, path)
    _validate_thumbnail_spec_value(spec)
    return spec


def _validate_thumbnail_spec_value(thumbnail_spec: Any) -> None:
    if not isinstance(thumbnail_spec, dict):
        raise ValueError("thumbnail_spec must be a JSON object (dict).")

    required_keys = {
        "image_format",
        "coord_space",
        "thumbnail_level",
        "downscale_x",
        "downscale_y",
    }
    missing_keys = sorted(required_keys - set(thumbnail_spec))
    if missing_keys:
        raise ValueError(f"thumbnail_spec missing required keys: {missing_keys}")

    if str(thumbnail_spec["coord_space"]) != "level0":
        raise ValueError(
            "thumbnail_spec.coord_space must be 'level0'. "
            f"Got {thumbnail_spec['coord_space']!r}."
        )

    try:
        int(thumbnail_spec["thumbnail_level"])
    except Exception as exc:
        raise ValueError("thumbnail_spec.thumbnail_level must be an int.") from exc

    for key in ("downscale_x", "downscale_y"):
        try:
            value = float(thumbnail_spec[key])
        except Exception as exc:
            raise ValueError(f"thumbnail_spec.{key} must be numeric.") from exc
        if value <= 0:
            raise ValueError(f"thumbnail_spec.{key} must be > 0. Got {value}.")
