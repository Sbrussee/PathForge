from __future__ import annotations

import numpy as np
import pytest

from pathforge.slide_retrieval.search_strategies.strategies.sish.sish_crops import (
    SISH_CANONICAL_TILE_MPP,
    SISH_CANONICAL_TILE_PX,
    read_canonical_sish_crop,
)


class _SlideProcessor:
    def __init__(self, base_mpp: float = 0.5) -> None:
        self.read_request: tuple[int, int, int, int, int] | None = None
        self.base_mpp = base_mpp

    def get_base_mpp(self, _wsi) -> float:
        return self.base_mpp

    def read_patch_region(self, _wsi, *, x, y, width, height, level):
        self.read_request = (x, y, width, height, level)
        return np.zeros((height, width, 3), dtype=np.uint8)


def test_canonical_sish_crop_is_centred_on_the_source_tile() -> None:
    processor = _SlideProcessor()

    crop = read_canonical_sish_crop(
        slide_processor=processor,
        wsi=object(),
        x_level0=100,
        y_level0=200,
        source_tile_px=256,
        source_tile_mpp=0.5,
    )

    assert processor.read_request == (-284, -184, 1024, 1024, 0)
    assert crop.shape == (SISH_CANONICAL_TILE_PX, SISH_CANONICAL_TILE_PX, 3)
    assert SISH_CANONICAL_TILE_MPP == 0.5


def test_canonical_sish_crop_downsamples_finer_source_imagery() -> None:
    processor = _SlideProcessor(base_mpp=0.25)

    crop = read_canonical_sish_crop(
        slide_processor=processor,
        wsi=object(),
        x_level0=0,
        y_level0=0,
        source_tile_px=256,
        source_tile_mpp=0.5,
    )

    assert processor.read_request == (-768, -768, 2048, 2048, 0)
    assert crop.shape == (1024, 1024, 3)


def test_canonical_sish_crop_rejects_source_imagery_coarser_than_target() -> None:
    with pytest.raises(ValueError, match="finer than 0.5 mpp"):
        read_canonical_sish_crop(
            slide_processor=_SlideProcessor(base_mpp=1.0),
            wsi=object(),
            x_level0=0,
            y_level0=0,
            source_tile_px=256,
            source_tile_mpp=0.5,
        )
