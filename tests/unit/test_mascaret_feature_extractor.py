"""Tests for the PathForge-native Mascaret feature extractor."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
import torch


def _mock_model() -> Mock:
    """Build a Mascaret-shaped model double with known normalization values."""
    model = Mock()
    model.config.pixel_mean = [0.25, 0.5, 0.75]
    model.config.pixel_std = [0.5, 0.25, 0.125]
    model.encode.return_value = torch.ones((2, 1536))
    return model


def test_mascaret_loads_default_remote_code_and_encodes_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mascaret loads its current repository default and returns CLS embeddings."""
    from pathforge.core.feature_extractors import mascaret

    model = _mock_model()
    load_model = Mock(return_value=model)
    monkeypatch.setattr(mascaret.AutoModel, "from_pretrained", load_model)

    extractor = mascaret.MascaretFeatureExtractor()
    embeddings = extractor.encode_images(torch.zeros((2, 3, 224, 224)))

    load_model.assert_called_once_with(
        mascaret.MASCARET_REPOSITORY_ID,
        trust_remote_code=True,
    )
    assert embeddings.shape == (2, 1536)
    assert torch.equal(embeddings, torch.ones((2, 1536)))


def test_mascaret_transform_accepts_canonical_numpy_patch_and_uses_model_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mascaret accepts PathForge RGB patches and uses its model normalization."""
    from pathforge.core.feature_extractors import mascaret

    model = _mock_model()
    monkeypatch.setattr(mascaret.AutoModel, "from_pretrained", Mock(return_value=model))
    extractor = mascaret.MascaretFeatureExtractor()

    patch = np.full((240, 300, 3), 255, dtype=np.uint8)
    transformed = extractor.get_transform()(patch)

    assert transformed.shape == (3, 224, 224)
    assert torch.allclose(
        transformed[:, 0, 0], torch.tensor([1.5, 2.0, 2.0]), atol=1e-6
    )


def test_mascaret_reports_gated_repository_access_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gated Hub failures tell users how to request access and authenticate."""
    from pathforge.core.feature_extractors import mascaret

    monkeypatch.setattr(
        mascaret.AutoModel,
        "from_pretrained",
        Mock(side_effect=OSError("403 Client Error: gated repo")),
    )

    with pytest.raises(RuntimeError, match="Accept Mascaret's access terms") as error:
        mascaret.MascaretFeatureExtractor()

    assert "hf auth login" in str(error.value)
    assert "HF_TOKEN" in str(error.value)


def test_mascaret_rejects_unexpected_embedding_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mascaret fails clearly if its pinned model stops returning CLS vectors."""
    from pathforge.core.feature_extractors import mascaret

    model = _mock_model()
    model.encode.return_value = torch.ones((2, 768))
    monkeypatch.setattr(mascaret.AutoModel, "from_pretrained", Mock(return_value=model))
    extractor = mascaret.MascaretFeatureExtractor()

    with pytest.raises(RuntimeError, match=r"\[B, 1536\]"):
        extractor.encode_images(torch.zeros((2, 3, 224, 224)))


def test_mascaret_is_registered_as_a_native_lazyslide_extractor() -> None:
    """Registry population exposes Mascaret for LazySlide adapter routing."""
    from pathforge.core.feature_extractors.factory import get_feature_extractor
    from pathforge.core.feature_extractors.mascaret import MascaretFeatureExtractor
    from pathforge.utils import registries

    assert "pathforge.core.feature_extractors.mascaret" in (
        registries._NATIVE_FEATURE_EXTRACTOR_MODULES
    )
    registries.populate_pathforge_feature_extractors()

    assert get_feature_extractor("mascaret") is MascaretFeatureExtractor
