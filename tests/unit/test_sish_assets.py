from __future__ import annotations

from types import SimpleNamespace

import pytest

from pathforge.slide_retrieval.search_strategies.strategies.sish import sish_assets
from pathforge.slide_retrieval.search_strategies.strategies.sish.sish_assets import (
    SISH_CODEBOOK_FILENAME,
    SISH_VQVAE_CHECKPOINT_FILENAME,
    load_sish_codebook,
    load_sish_vqvae_encoder,
    require_sish_asset_path,
    resolve_sish_asset_path,
    validate_sish_codebook,
)


def test_sish_assets_resolve_from_the_configured_weights_directory(tmp_path) -> None:
    config = SimpleNamespace(
        slide_retrieval=SimpleNamespace(weights_dir=tmp_path),
    )

    assert (
        resolve_sish_asset_path(config=config, asset="vqvae_checkpoint")
        == tmp_path / SISH_VQVAE_CHECKPOINT_FILENAME
    )
    assert (
        resolve_sish_asset_path(config=config, asset="codebook_semantic")
        == tmp_path / SISH_CODEBOOK_FILENAME
    )


def test_sish_assets_preserve_explicit_checkpoint_override(tmp_path) -> None:
    override = tmp_path / "custom-vqvae.pth"
    config = SimpleNamespace(
        slide_retrieval=SimpleNamespace(weights_dir=tmp_path / "default"),
        sish=SimpleNamespace(vqvae_checkpoint=override),
    )

    assert resolve_sish_asset_path(config=config, asset="vqvae_checkpoint") == override


def test_sish_assets_explain_how_to_fix_a_missing_required_file(tmp_path) -> None:
    config = SimpleNamespace(slide_retrieval=SimpleNamespace(weights_dir=tmp_path))

    with pytest.raises(FileNotFoundError, match="slide_retrieval.weights_dir"):
        require_sish_asset_path(config=config, asset="codebook_semantic")


def test_sish_codebook_validation_normalizes_a_complete_mapping() -> None:
    codebook = validate_sish_codebook({code: str(code + 1) for code in range(128)})

    assert codebook[0] == 1
    assert codebook[127] == 128


def test_sish_codebook_validation_rejects_a_missing_code() -> None:
    with pytest.raises(ValueError, match="Missing code 127"):
        validate_sish_codebook({code: code for code in range(127)})


def test_sish_vqvae_loader_loads_a_compatible_checkpoint(tmp_path, monkeypatch) -> None:
    checkpoint_path = tmp_path / SISH_VQVAE_CHECKPOINT_FILENAME
    checkpoint_path.touch()
    config = SimpleNamespace(slide_retrieval=SimpleNamespace(weights_dir=tmp_path))
    fake_model = _FakeVQVAE()
    monkeypatch.setattr(
        sish_assets,
        "LargeVectorQuantizedVAE_Encode",
        lambda **_: fake_model,
    )
    monkeypatch.setattr(
        sish_assets.torch,
        "load",
        lambda *_args, **_kwargs: {"model": {"module.encoder.0.weight": "weight"}},
    )

    loaded = load_sish_vqvae_encoder(
        config=config,
        device=sish_assets.torch.device("cpu"),
    )

    assert loaded is fake_model
    assert fake_model.loaded_weights == {"encoder.0.weight": "weight"}
    assert fake_model.is_eval is True


def test_sish_vqvae_loader_explains_an_incompatible_checkpoint(
    tmp_path, monkeypatch
) -> None:
    checkpoint_path = tmp_path / SISH_VQVAE_CHECKPOINT_FILENAME
    checkpoint_path.touch()
    config = SimpleNamespace(slide_retrieval=SimpleNamespace(weights_dir=tmp_path))
    monkeypatch.setattr(
        sish_assets.torch,
        "load",
        lambda *_args, **_kwargs: {"not_model": {}},
    )

    with pytest.raises(ValueError, match="compatible SISH VQ-VAE checkpoint"):
        load_sish_vqvae_encoder(config=config, device=sish_assets.torch.device("cpu"))


def test_sish_codebook_loader_explains_an_incompatible_codebook(
    tmp_path, monkeypatch
) -> None:
    codebook_path = tmp_path / SISH_CODEBOOK_FILENAME
    codebook_path.touch()
    config = SimpleNamespace(slide_retrieval=SimpleNamespace(weights_dir=tmp_path))
    monkeypatch.setattr(
        sish_assets.torch,
        "load",
        lambda *_args, **_kwargs: {0: 0},
    )

    with pytest.raises(ValueError, match="compatible SISH codebook"):
        load_sish_codebook(config=config)


class _FakeVQVAE:
    def __init__(self) -> None:
        self.loaded_weights = None
        self.is_eval = False

    def load_state_dict(self, weights, *, strict: bool) -> None:
        assert strict is True
        self.loaded_weights = weights

    def to(self, _device):
        return self

    def eval(self):
        self.is_eval = True
        return self
