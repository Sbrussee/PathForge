"""Unit tests for stain / color normalization in feature extraction.

Color normalization is configured via the ``color_norm`` benchmark parameter
(``macenko`` / ``reinhard``), folded into the stored feature name, and applied
during feature extraction by the LazySlide patch — whose ``_restore_image_like_tile``
helper converts the normalized tile back to the image-like ``HWC`` ``uint8``
format the downstream transform expects.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathforge.config.config import BenchmarkParameters
from pathforge.core.experiments.combinations import ComboConfig
from pathforge.core.experiments.combo_ids import build_feature_name

_BASE_BENCH = {
    "tile_px": [256],
    "tile_mpp": [0.5],
    "feature_extraction": ["resnet18"],
    "mil": [],
}


class TestColorNormConfigValidation:
    @pytest.mark.parametrize("value", ["macenko", "reinhard", "MACENKO", "Reinhard"])
    def test_valid_color_norm_accepted(self, value: str) -> None:
        params = BenchmarkParameters.model_validate(
            {**_BASE_BENCH, "color_norm": [value]}
        )
        assert params.color_norm == [value]

    def test_none_color_norm_allowed(self) -> None:
        params = BenchmarkParameters.model_validate(
            {**_BASE_BENCH, "color_norm": None}
        )
        assert params.color_norm is None

    @pytest.mark.parametrize("value", ["vahadane", "bogus", "histogram"])
    def test_invalid_color_norm_rejected(self, value: str) -> None:
        with pytest.raises(Exception) as error:
            BenchmarkParameters.model_validate({**_BASE_BENCH, "color_norm": [value]})
        assert "color_norm" in str(error.value)


class TestColorNormFeatureName:
    def test_feature_name_includes_color_norm(self) -> None:
        combo = ComboConfig.from_keys_values(
            ["feature_extraction", "color_norm"], ["resnet18", "macenko"]
        )
        assert build_feature_name(combo) == "resnet18_macenko"

    def test_feature_name_without_color_norm(self) -> None:
        combo = ComboConfig.from_keys_values(["feature_extraction"], ["resnet18"])
        assert build_feature_name(combo) == "resnet18"

    def test_feature_name_strips_blank_color_norm(self) -> None:
        combo = ComboConfig.from_keys_values(
            ["feature_extraction", "color_norm"], ["resnet18", "   "]
        )
        assert build_feature_name(combo) == "resnet18"


class TestRestoreImageLikeTile:
    """The color-normalized tile must come back as HWC uint8 for the transform."""

    @pytest.fixture(autouse=True)
    def _require_lazyslide(self) -> None:
        pytest.importorskip("lazyslide")
        pytest.importorskip("torch")

    @staticmethod
    def _restore():
        from pathforge.core.slide_processing.lazyslide_patch import (
            _restore_image_like_tile,
        )

        return _restore_image_like_tile

    def test_torch_chw_float_becomes_hwc_uint8_and_clamps(self) -> None:
        import torch

        tile = torch.zeros(3, 4, 5, dtype=torch.float32)
        tile[0] = 300.0  # over-range -> clamps to 255
        tile[1] = -20.0  # under-range -> clamps to 0
        tile[2] = 128.0

        out = self._restore()(tile)

        assert isinstance(out, np.ndarray)
        assert out.shape == (4, 5, 3)  # CHW -> HWC
        assert out.dtype == np.uint8
        assert out[..., 0].max() == 255
        assert out[..., 1].min() == 0
        assert int(out[0, 0, 2]) == 128

    def test_torch_hwc_uint8_passthrough(self) -> None:
        import torch

        tile = (torch.ones(4, 5, 3) * 77).to(torch.uint8)
        out = self._restore()(tile)

        assert isinstance(out, np.ndarray)
        assert out.shape == (4, 5, 3)
        assert out.dtype == np.uint8
        assert int(out[0, 0, 0]) == 77

    def test_numpy_chw_float_becomes_hwc_uint8(self) -> None:
        tile = np.full((3, 4, 5), 200.0, dtype=np.float32)
        out = self._restore()(tile)

        assert out.shape == (4, 5, 3)
        assert out.dtype == np.uint8
        assert int(out[0, 0, 0]) == 200

    def test_numpy_hwc_uint8_unchanged(self) -> None:
        tile = np.full((4, 5, 3), 42, dtype=np.uint8)
        out = self._restore()(tile)

        assert np.array_equal(out, tile)
        assert out.dtype == np.uint8
