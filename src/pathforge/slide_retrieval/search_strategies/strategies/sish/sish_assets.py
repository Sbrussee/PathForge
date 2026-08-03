"""Resolution of user-supplied SISH model assets.

The default layout keeps downloaded weights out of configuration files while
allowing callers to override individual legacy SISH asset paths.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from pathforge.slide_retrieval.search_strategies.strategies.sish.sish_vqvae import (
    LargeVectorQuantizedVAE_Encode,
)

SISH_VQVAE_CHECKPOINT_FILENAME = "sish_vqvae_checkpoint.pth"
SISH_CODEBOOK_FILENAME = "sish_codebook.pt"
SISH_TRASH_CLASSIFIER_FILENAME = "sish_trash_classifier.pkl"
DEFAULT_SLIDE_RETRIEVAL_WEIGHTS_DIR = Path("model_weights/slide_retrieval")

_ASSET_FILENAMES = {
    "vqvae_checkpoint": SISH_VQVAE_CHECKPOINT_FILENAME,
    "codebook_semantic": SISH_CODEBOOK_FILENAME,
    "trash_classifier": SISH_TRASH_CLASSIFIER_FILENAME,
}
_LEGACY_PATHS = {
    "vqvae_checkpoint": [
        ("experiment", "sish", "vqvae_checkpoint"),
        ("experiment", "SISH_metrics", "vqvae_checkpoint"),
        ("sish", "vqvae_checkpoint"),
    ],
    "codebook_semantic": [
        ("experiment", "sish", "codebook_semantic"),
        ("experiment", "SISH_metrics", "codebook_semantic"),
        ("sish", "codebook_semantic"),
    ],
    "trash_classifier": [
        ("experiment", "sish", "trash_classifier"),
        ("experiment", "SISH_metrics", "trash_classifier"),
        ("sish", "trash_classifier"),
    ],
}


def resolve_sish_asset_path(*, config: Any, asset: str) -> Path:
    """Resolve the configured or conventional path for one SISH asset.

    Inputs:
        config: PathForge config object or mapping.
        asset: ``"vqvae_checkpoint"``, ``"codebook_semantic"``, or
            ``"trash_classifier"``.

    Output:
        The path selected by an explicit legacy override, or by
        ``slide_retrieval.weights_dir`` plus the standard asset filename.

    Example:
        ``resolve_sish_asset_path(config=cfg, asset="codebook_semantic")``.
    """
    if asset not in _ASSET_FILENAMES:
        raise ValueError(f"Unknown SISH asset: {asset!r}.")

    override = _get_config_value(config, _LEGACY_PATHS.get(asset, []))
    if override is not None:
        return Path(override)

    weights_dir = _get_config_value(config, [("slide_retrieval", "weights_dir")])
    return (
        Path(weights_dir or DEFAULT_SLIDE_RETRIEVAL_WEIGHTS_DIR)
        / _ASSET_FILENAMES[asset]
    )


def require_sish_asset_path(*, config: Any, asset: str) -> Path:
    """Return a required SISH asset path or raise an actionable missing-file error."""
    path = resolve_sish_asset_path(config=config, asset=asset)
    if path.is_file():
        return path

    raise FileNotFoundError(
        f"Missing SISH {asset.replace('_', ' ')} at {path}. Download the compatible "
        "asset into slide_retrieval.weights_dir or set the legacy explicit SISH "
        "path override. See docs/slide-retrieval.md#sish-model-assets."
    )


def configured_sish_asset_path(*, config: Any, asset: str) -> Path | None:
    """Return an asset path only when the supplied config declares its location.

    This keeps cache contracts compatible with lightweight legacy config objects
    that do not carry a ``slide_retrieval`` section.
    """
    if asset not in _ASSET_FILENAMES:
        raise ValueError(f"Unknown SISH asset: {asset!r}.")

    override = _get_config_value(config, _LEGACY_PATHS.get(asset, []))
    if override is not None:
        return Path(override)

    weights_dir = _get_config_value(config, [("slide_retrieval", "weights_dir")])
    if weights_dir is None:
        return None
    return Path(weights_dir) / _ASSET_FILENAMES[asset]


def validate_sish_codebook(codebook: Any) -> dict[int, int]:
    """Validate the semantic mapping expected by the 128-entry SISH VQ-VAE.

    Inputs:
        codebook: Deserialized mapping from raw VQ-VAE code to semantic code.

    Output:
        A normalized mapping for every raw code in ``[0, 127]``.

    Example:
        ``validate_sish_codebook({code: code for code in range(128)})``.
    """
    if not isinstance(codebook, Mapping):
        raise ValueError("SISH codebook must deserialize to a mapping.")

    normalized: dict[int, int] = {}
    for raw_code in range(128):
        if raw_code not in codebook:
            raise ValueError(
                "SISH codebook must provide semantic values for raw codes 0 through 127. "
                f"Missing code {raw_code}."
            )
        try:
            normalized[raw_code] = int(codebook[raw_code])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"SISH codebook semantic value for raw code {raw_code} must be an integer."
            ) from exc
    return normalized


def load_sish_codebook(*, config: Any) -> dict[int, int]:
    """Load and validate the configured SISH semantic codebook.

    Inputs:
        config: PathForge config object or mapping.

    Output:
        Semantic mapping for all 128 raw VQ-VAE codes.

    Example:
        ``codebook = load_sish_codebook(config=cfg)``.
    """
    path = require_sish_asset_path(config=config, asset="codebook_semantic")
    try:
        return validate_sish_codebook(torch.load(path, map_location="cpu"))
    except Exception as exc:
        raise ValueError(
            f"Could not load a compatible SISH codebook from {path}."
        ) from exc


def load_sish_vqvae_encoder(*, config: Any, device: torch.device) -> torch.nn.Module:
    """Load the supported SISH encoder from a user-supplied checkpoint.

    Inputs:
        config: PathForge config object or mapping.
        device: Device to host the evaluation-only encoder.

    Output:
        Loaded ``LargeVectorQuantizedVAE_Encode`` in evaluation mode.

    Example:
        ``encoder = load_sish_vqvae_encoder(config=cfg, device=torch.device("cpu"))``.
    """
    path = require_sish_asset_path(config=config, asset="vqvae_checkpoint")
    try:
        checkpoint = torch.load(path, map_location="cpu")
        if not isinstance(checkpoint, Mapping) or not isinstance(
            checkpoint.get("model"), Mapping
        ):
            raise ValueError("checkpoint does not contain a mapping at key 'model'.")
        encoder_weights = {
            key[len("module.") :]: value
            for key, value in checkpoint["model"].items()
            if key.startswith(("module.encoder.", "module.codebook."))
        }
        model = LargeVectorQuantizedVAE_Encode(code_dim=256, code_size=128)
        model.load_state_dict(encoder_weights, strict=True)
    except Exception as exc:
        raise ValueError(
            f"Could not load a compatible SISH VQ-VAE checkpoint from {path}."
        ) from exc
    return model.to(device).eval()


def load_sish_trash_classifier(*, config: Any) -> Any:
    """Load the upstream-compatible SISH LBP trash classifier.

    The returned object must implement ``predict`` over a ``(N, 128)`` LBP
    histogram matrix.
    """
    path = require_sish_asset_path(config=config, asset="trash_classifier")
    try:
        with path.open("rb") as handle:
            classifier = pickle.load(handle)
    except Exception as exc:
        raise ValueError(
            f"Could not load a compatible SISH trash classifier from {path}."
        ) from exc
    if not callable(getattr(classifier, "predict", None)):
        raise ValueError(
            f"Could not load a compatible SISH trash classifier from {path}."
        )
    return classifier


def _get_config_value(config: Any, candidate_paths: list[tuple[str, ...]]) -> Any:
    """Return the first non-null nested value from a mapping or config object."""
    for path in candidate_paths:
        current = config
        for key in path:
            if current is None:
                break
            current = (
                current.get(key)
                if isinstance(current, dict)
                else getattr(current, key, None)
            )
        if current is not None:
            return current
    return None
