from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pathbench.core.io.h5.base import FileHandleH5
from pathbench.core.io.h5 import heatmaps as heatmap_io
from pathbench.core.io.h5 import tiles as tiles_io
from pathbench.utils.optional.torchmil import require_torchmil
from pathbench.utils.registries import EXPLAINERS, populate_dynamic_registries


@dataclass(frozen=True)
class InferenceHeatmapResult:
    """Summary of a heatmap produced during inference.

    Attributes:
        artifact_path: H5 artifact that received the heatmap.
        bag_id: Bag id inside the slide artifact, for example
            ``"256px_0.5mpp"``.
        heatmap_name: Prediction heatmap namespace.
        num_points: Number of unmasked score/coordinate pairs persisted.
        output_path: Optional JSON sidecar path written for downstream tools.
    """

    artifact_path: Path
    bag_id: str
    heatmap_name: str
    num_points: int
    output_path: Path | None = None


def create_inference_heatmap(
    *,
    artifact_path: str | Path,
    bag_id: str,
    scores_path: str | Path,
    heatmap_backend: str = "torchmil",
    heatmap_name: str = "torchmil",
    output_path: str | Path | None = None,
    coords_path: str | Path | None = None,
    mask_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> InferenceHeatmapResult:
    """Create and persist an inference heatmap through an explainer registry.

    Args:
        artifact_path: Slide H5 artifact containing `bags/{bag_id}/coords`, or
            receiving the heatmap when `coords_path` is provided.
        bag_id: Bag id whose coordinates align with the prediction scores.
        scores_path: `.npy`, `.npz`, or JSON file containing one score per
            instance shaped `[N]`.
        heatmap_backend: Explainer backend key. `"torchmil"` maps to the
            registry key `"torchmil_heatmap"`.
        heatmap_name: H5 namespace under
            `bags/{bag_id}/predictions/heatmaps/{heatmap_name}`.
        output_path: Optional JSON sidecar path containing coordinates and
            scores for non-H5 consumers.
        coords_path: Optional `.npy`, `.npz`, or JSON coordinate file shaped
            `[N, 2]`. If omitted, coords are read from the H5 bag coords and the
            first two columns are used.
        mask_path: Optional `.npy`, `.npz`, or JSON boolean/binary mask shaped
            `[N]`.
        model_path: Optional model checkpoint path recorded in heatmap metadata.

    Returns:
        InferenceHeatmapResult: Summary of the persisted heatmap.

    Example:
        ```python
        result = create_inference_heatmap(
            artifact_path="SLIDE_001.h5",
            bag_id="256px_0.5mpp",
            scores_path="attention_scores.npy",
            heatmap_backend="torchmil",
            heatmap_name="abmil_attention",
        )
        assert result.num_points > 0
        ```

    Raises:
        KeyError: If the selected explainer backend is not registered.
        ValueError: If score/coordinate shapes are invalid or row counts differ.
        RuntimeError: If the selected optional backend is unavailable.
    """

    artifact = Path(artifact_path)
    scores = _load_array(scores_path, expected_ndim=1, name="scores").astype(np.float32, copy=False)
    coords = (
        _load_array(coords_path, expected_ndim=2, name="coords")
        if coords_path is not None
        else _read_xy_coords_from_h5(artifact, bag_id)
    )
    mask = (
        _load_array(mask_path, expected_ndim=1, name="mask").astype(bool, copy=False)
        if mask_path is not None
        else None
    )

    if coords.shape[1] < 2:
        raise ValueError(f"coords must have at least two columns. Got {coords.shape}.")
    coords_xy = coords[:, :2]
    if coords_xy.shape[0] != scores.shape[0]:
        raise ValueError(
            "coords and scores must have the same row count before masking. "
            f"Got {coords_xy.shape[0]} and {scores.shape[0]}."
        )
    if mask is not None and mask.shape[0] != scores.shape[0]:
        raise ValueError(
            f"mask must have one value per score. Got {mask.shape[0]} mask values and {scores.shape[0]} scores."
        )

    explainer_key = _resolve_heatmap_explainer_key(heatmap_backend)
    if heatmap_backend == "torchmil":
        require_torchmil("Inference heatmap backend 'torchmil'")
    populate_dynamic_registries()
    ExplainerClass = EXPLAINERS.get(explainer_key)
    explainer = ExplainerClass()
    explainer.initialize(
        {
            "artifact_path": str(artifact),
            "bag_id": bag_id,
            "heatmap_backend": heatmap_backend,
            "heatmap_name": heatmap_name,
        }
    )

    payload: dict[str, torch.Tensor] = {
        "coords": torch.as_tensor(coords_xy, dtype=torch.float32),
        "instance_scores": torch.as_tensor(scores, dtype=torch.float32),
    }
    if mask is not None:
        payload["mask"] = torch.as_tensor(mask, dtype=torch.bool)

    heatmap = explainer.explain(payload)
    heatmap_coords = _tensor_to_numpy(heatmap.coords)
    heatmap_scores = _tensor_to_numpy(heatmap.scores).astype(np.float32, copy=False)
    metadata = {
        "backend": heatmap_backend,
        "explainer": explainer_key,
        "model_path": str(model_path) if model_path is not None else None,
        "scores_path": str(scores_path),
        "coords_path": str(coords_path) if coords_path is not None else None,
        "mask_path": str(mask_path) if mask_path is not None else None,
        "score_range": [float(heatmap_scores.min()), float(heatmap_scores.max())],
        "coord_space": "level0_xy",
    }

    with FileHandleH5(artifact, mode="a") as slide_artifact:
        heatmap_io.write_prediction_heatmap(
            slide_artifact,
            bag_id,
            heatmap_name,
            coords=heatmap_coords,
            scores=heatmap_scores,
            metadata=metadata,
        )

    out_path = Path(output_path) if output_path is not None else None
    if out_path is not None:
        _write_json_sidecar(out_path, coords=heatmap_coords, scores=heatmap_scores, metadata=metadata)

    return InferenceHeatmapResult(
        artifact_path=artifact,
        bag_id=bag_id,
        heatmap_name=heatmap_name,
        num_points=int(heatmap_scores.shape[0]),
        output_path=out_path,
    )


def _resolve_heatmap_explainer_key(heatmap_backend: str) -> str:
    if heatmap_backend == "torchmil":
        return "torchmil_heatmap"
    return heatmap_backend


def _read_xy_coords_from_h5(artifact_path: Path, bag_id: str) -> np.ndarray:
    with FileHandleH5(artifact_path, mode="r") as slide_artifact:
        coords = tiles_io.read_coords(slide_artifact, bag_id)
    return coords[:, :2].astype(np.float32, copy=False)


def _load_array(path: str | Path | None, *, expected_ndim: int, name: str) -> np.ndarray:
    if path is None:
        raise ValueError(f"{name}_path is required.")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} file does not exist: {p}")
    if p.suffix == ".npy":
        arr = np.load(p)
    elif p.suffix == ".npz":
        loaded = np.load(p)
        first_key = loaded.files[0]
        arr = loaded[first_key]
    elif p.suffix == ".json":
        import json

        arr = np.asarray(json.loads(p.read_text(encoding="utf-8")))
    else:
        raise ValueError(f"{name} file must be .npy, .npz, or .json. Got {p.suffix!r}.")
    if arr.ndim != expected_ndim:
        raise ValueError(f"{name} must have rank {expected_ndim}. Got shape {arr.shape}.")
    if not np.isfinite(arr.astype(np.float32, copy=False)).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return arr


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _write_json_sidecar(
    output_path: Path,
    *,
    coords: np.ndarray,
    scores: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "coords": coords.tolist(),
        "scores": scores.tolist(),
        "metadata": metadata,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
