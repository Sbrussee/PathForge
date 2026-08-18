"""Tests for the PathForge-native Phaet feature extractor."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
import torch


def _mock_model() -> Mock:
    """Build a Phaet-shaped model double with known normalization values."""
    model = Mock()
    model.config.pixel_mean = [0.485, 0.456, 0.406]
    model.config.pixel_std = [0.229, 0.224, 0.225]
    model.encode.return_value = torch.ones((2, 1024))
    return model


def test_phaet_loads_remote_code_and_encodes_cls_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phaet loads its supported model interface and returns CLS embeddings."""
    from pathforge.core.feature_extractors import phaet

    model = _mock_model()
    load_model = Mock(return_value=model)
    monkeypatch.setattr(phaet.AutoModel, "from_pretrained", load_model)

    extractor = phaet.PhaetFeatureExtractor()
    embeddings = extractor.encode_images(torch.zeros((2, 3, 224, 224)))

    load_model.assert_called_once_with(
        phaet.PHAET_REPOSITORY_ID,
        trust_remote_code=True,
    )
    assert embeddings.shape == (2, 1024)


def test_phaet_transform_accepts_canonical_numpy_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phaet preprocesses the canonical PathForge RGB NumPy patch."""
    from pathforge.core.feature_extractors import phaet

    monkeypatch.setattr(phaet.AutoModel, "from_pretrained", Mock(return_value=_mock_model()))
    extractor = phaet.PhaetFeatureExtractor()

    transformed = extractor.get_transform()(np.full((240, 300, 3), 255, dtype=np.uint8))

    assert transformed.shape == (3, 224, 224)
    assert torch.allclose(
        transformed[:, 0, 0],
        torch.tensor([2.2489, 2.4286, 2.6400]),
        atol=1e-4,
    )


def test_phaet_reports_gated_repository_access_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phaet explains how to obtain gated model access."""
    from pathforge.core.feature_extractors import phaet

    monkeypatch.setattr(
        phaet.AutoModel,
        "from_pretrained",
        Mock(side_effect=OSError("403 Client Error: gated repo")),
    )

    with pytest.raises(RuntimeError, match="Accept Phaet's access terms"):
        phaet.PhaetFeatureExtractor()


def test_phaet_is_registered_as_a_native_lazyslide_extractor() -> None:
    """Registry population exposes Phaet for LazySlide adapter routing."""
    from pathforge.core.feature_extractors.factory import get_feature_extractor
    from pathforge.core.feature_extractors.phaet import PhaetFeatureExtractor
    from pathforge.utils import registries

    assert "pathforge.core.feature_extractors.phaet" in (
        registries._NATIVE_FEATURE_EXTRACTOR_MODULES
    )
    registries.populate_pathforge_feature_extractors()

    assert get_feature_extractor("phaet") is PhaetFeatureExtractor
