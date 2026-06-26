from __future__ import annotations

import copy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pathforge.adapters.torchmil.task_output import normalize_torchmil_output
from pathforge.config.config import Config
from pathforge.core.io.h5.base import FileHandleH5
from pathforge.core.io.h5.features import read_features
from pathforge.core.models.mil_base import MILModelBase
from pathforge.policy.utils import build_mil_model_for_config
from pathforge.utils.registries import populate_dynamic_registries


MODEL_PACKAGE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class LoadedModelPackage:
    """Self-contained packaged PathForge MIL model.

    Attributes:
        model: Reconstructed MIL model with weights loaded and set to eval mode.
        config: Validated config required to rebuild the architecture.
        package_payload: Raw serialized package dictionary loaded from disk.

    Example:
        ```python
        loaded = load_packaged_model("best_model_package.pt")
        logits = predict_bag(loaded.model, torch.randn(1, 8, 16), task=loaded.task)
        ```
    """

    model: MILModelBase
    config: Config
    package_payload: dict[str, Any]

    @property
    def task(self) -> str:
        """Return the PathForge task name stored in the package metadata."""

        return str(self.package_payload["task"])

    @property
    def model_name(self) -> str:
        """Return the registry model name stored in the package metadata."""

        return str(self.package_payload["model_name"])

    @property
    def inference_metadata(self) -> dict[str, Any]:
        """Return packaged feature-selection metadata for inference."""

        metadata = self.package_payload.get("inference_metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("Packaged inference metadata must be a dictionary.")
        return metadata


def save_packaged_model(
    *,
    path: str | Path,
    model: MILModelBase,
    config: Config,
    model_name: str,
    input_dim: int,
    output_dim: int,
    checkpoint_path: str | Path | None = None,
    loss_name: str | None = None,
) -> Path:
    """Serialize one self-contained inference package to disk.

    Args:
        path: Output ``.pt`` path.
        model: Trained MIL model whose weights will be saved.
        config: Active PathForge config used to reconstruct the architecture.
        model_name: Registry key used to instantiate the trained model.
        input_dim: Instance feature dimension ``D`` for bags shaped ``[B, N, D]``.
        output_dim: Number of output channels produced by the model.
        checkpoint_path: Optional originating Lightning checkpoint path.
        loss_name: Optional registry loss name for traceability.

    Returns:
        Path: Resolved output package path.
    """

    assert input_dim > 0, "input_dim must be positive."
    assert output_dim > 0, "output_dim must be positive."

    package_path = Path(path).resolve()
    package_path.parent.mkdir(parents=True, exist_ok=True)

    model_snapshot = copy.deepcopy(model).cpu().eval()
    model_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    payload = {
        "format": "pathforge_model_package",
        "format_version": MODEL_PACKAGE_FORMAT_VERSION,
        "pathforge_version": _pathforge_version(),
        "task": str(config.experiment.task or "classification"),
        "model_name": str(model_name),
        "loss_name": str(loss_name) if loss_name is not None else None,
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "checkpoint_path": str(Path(checkpoint_path).resolve()) if checkpoint_path else None,
        "config": config.model_dump(mode="python"),
        "inference_metadata": _build_inference_metadata(config),
        "model_object": model_snapshot,
        "state_dict": model_state,
    }
    torch.save(payload, package_path)
    return package_path


def load_packaged_model(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> LoadedModelPackage:
    """Load one packaged PathForge MIL model from disk.

    Args:
        path: Packaged model ``.pt`` path written by :func:`save_packaged_model`.
        map_location: Device mapping used when reading the serialized tensors.

    Returns:
        LoadedModelPackage: Reconstructed model plus validated config and metadata.
    """

    package_path = Path(path).resolve()
    payload = torch.load(package_path, map_location=map_location, weights_only=False)
    _validate_model_package_payload(payload, package_path)
    populate_dynamic_registries()
    config = Config.model_validate(payload["config"])
    model_object = payload.get("model_object")
    if isinstance(model_object, MILModelBase):
        model = model_object
    else:
        model = build_mil_model_for_config(
            config,
            model_name=str(payload["model_name"]),
            input_dim=int(payload["input_dim"]),
            output_dim=int(payload["output_dim"]),
        )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return LoadedModelPackage(
        model=model,
        config=config,
        package_payload=payload,
    )


def predict_bag(
    model: MILModelBase,
    bag: torch.Tensor,
    *,
    task: str,
    mask: torch.Tensor | None = None,
    coords: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one MIL forward pass and normalize the output tensor.

    Args:
        model: Loaded MIL model.
        bag: Feature tensor shaped ``[B, N, D]``.
        task: PathForge task name controlling output normalization.
        mask: Optional padding mask shaped ``[B, N]``.
        coords: Optional coordinates shaped ``[B, N, 2]``.

    Returns:
        torch.Tensor: Normalized prediction tensor.
    """

    assert bag.ndim == 3, f"Expected bag shape [B, N, D]. Got {tuple(bag.shape)}."
    with torch.no_grad():
        output = model.forward_bag(bag, mask=mask, coords=coords)
    if isinstance(output, dict) and "logits" in output and isinstance(output["logits"], torch.Tensor):
        tensor = output["logits"].float()
        if task in {"survival", "survival_discrete"}:
            return normalize_torchmil_output(output, task=task)
        return tensor
    return normalize_torchmil_output(output, task=task)


def load_bag_from_input(
    input_path: str | Path,
    *,
    bag_id: str | None = None,
    extractor_name: str | None = None,
) -> torch.Tensor:
    """Load one feature bag from ``.h5``, ``.pt``, ``.npy``, or ``.npz`` input.

    Args:
        input_path: Path to one feature artifact.
        bag_id: Required H5 bag identifier such as ``256px_0.5mpp``.
        extractor_name: Required H5 extractor key such as ``resnet50``.

    Returns:
        torch.Tensor: Batched feature tensor shaped ``[1, N, D]``.
    """

    feature_path = Path(input_path).resolve()
    suffix = feature_path.suffix.lower()
    if suffix == ".h5":
        if bag_id is None or extractor_name is None:
            raise ValueError(
                "H5 inference requires both bag_id and extractor_name."
            )
        with FileHandleH5(feature_path, mode="r") as handle:
            features = read_features(
                handle,
                bag_id=bag_id,
                extractor_name=extractor_name,
            )
        tensor = torch.from_numpy(features)
    elif suffix == ".pt":
        tensor = torch.load(feature_path, map_location="cpu", weights_only=False)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(".pt inference inputs must deserialize to a torch.Tensor.")
    elif suffix in {".npy", ".npz"}:
        array = np.load(feature_path)
        if isinstance(array, np.lib.npyio.NpzFile):
            if len(array.files) != 1:
                raise ValueError(".npz inference inputs must contain exactly one array.")
            array = array[array.files[0]]
        tensor = torch.from_numpy(np.asarray(array))
    else:
        raise ValueError(
            f"Unsupported inference input format {suffix!r}. Expected .h5, .pt, .npy, or .npz."
        )

    tensor = tensor.float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(
            f"Inference bags must have shape [N, D] or [B, N, D]. Got {tuple(tensor.shape)}."
        )
    return tensor


def package_path_from_checkpoint(checkpoint_path: str | Path) -> Path:
    """Return the default packaged-model path next to one Lightning checkpoint."""

    checkpoint = Path(checkpoint_path).resolve()
    return checkpoint.with_name(f"{checkpoint.stem}_package.pt")


def select_inference_feature_metadata(
    package: LoadedModelPackage,
    *,
    bag_id: str | None,
    extractor_name: str | None,
) -> tuple[str | None, str | None]:
    """Resolve bag/extractor selection from CLI overrides or packaged defaults."""

    metadata = package.inference_metadata
    resolved_bag_id = bag_id or _optional_str(metadata.get("bag_id"))
    resolved_extractor = extractor_name or _optional_str(metadata.get("feature_extractor"))
    return resolved_bag_id, resolved_extractor


def _build_inference_metadata(config: Config) -> dict[str, Any]:
    """Build packaged feature-selection metadata for inference-time defaults.

    Args:
        config: PathForge config used to infer the active bag source and
            feature extractor selection.

    Returns:
        dict[str, Any]: JSON-serializable metadata containing optional
        ``bag_id``, ``feature_extractor``, ``tile_px``, and ``tile_mpp``.
    """

    search_params = getattr(config, "_active_search_params", {})
    if not isinstance(search_params, dict):
        search_params = {}

    tile_px = search_params.get("tile_px")
    if tile_px is None and config.benchmark_parameters.tile_px:
        tile_px = config.benchmark_parameters.tile_px[0]

    tile_mpp = search_params.get("tile_mpp")
    if tile_mpp is None and config.benchmark_parameters.tile_mpp:
        tile_mpp = config.benchmark_parameters.tile_mpp[0]

    feature_extractor = search_params.get("feature_extraction")
    if feature_extractor is None and config.benchmark_parameters.feature_extraction:
        feature_extractor = config.benchmark_parameters.feature_extraction[0]

    bag_id = None
    if tile_px is not None and tile_mpp is not None:
        bag_id = f"{int(tile_px)}px_{tile_mpp}mpp"

    return {
        "bag_id": bag_id,
        "feature_extractor": str(feature_extractor) if feature_extractor is not None else None,
        "tile_px": int(tile_px) if tile_px is not None else None,
        "tile_mpp": float(tile_mpp) if tile_mpp is not None else None,
    }


def _optional_str(value: Any) -> str | None:
    """Convert an optional value to string while preserving ``None``."""

    if value is None:
        return None
    return str(value)


def _pathforge_version() -> str:
    """Return the installed PathForge package version when available."""

    try:
        return version("pathforge")
    except PackageNotFoundError:
        return "unknown"


def _validate_model_package_payload(payload: Any, path: Path) -> None:
    """Validate the structural invariants of a serialized model package.

    Args:
        payload: Object deserialized from ``torch.load``.
        path: Package path used for error reporting.

    Raises:
        TypeError: If the payload is not dictionary-like.
        ValueError: If the payload format marker is invalid.
        KeyError: If required fields are missing.
    """

    if not isinstance(payload, dict):
        raise TypeError(f"Model package at {path} must deserialize to a dict.")
    if payload.get("format") != "pathforge_model_package":
        raise ValueError(f"{path} is not a PathForge packaged model artifact.")
    required_keys = {
        "config",
        "format_version",
        "input_dim",
        "model_name",
        "output_dim",
        "state_dict",
        "task",
    }
    missing = sorted(required_keys - set(payload))
    if missing:
        raise KeyError(f"Model package at {path} is missing keys: {missing}.")
