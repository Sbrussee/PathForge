from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pathbench.core.io.h5.base import FileHandleH5
from pathbench.core.io.h5 import heatmaps as heatmap_io
from pathbench.core.io.h5 import tiles as tiles_io
from pathbench.core.io.h5.layout import DEFAULT_LAYOUT
from pathbench.utils.optional.torchmil import require_torchmil
from pathbench.utils.registries import EXPLAINERS, populate_dynamic_registries

logger = logging.getLogger(__name__)


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
        image_output_path: Optional PNG visualization path for quick inspection.
        smoothed_image_output_path: Optional PNG path containing a blurred
            "cloud" view of the same heatmap.
        top_tiles_output_path: Optional PNG path containing a top-K tiles table
            with per-tile scores and coordinates.
    """

    artifact_path: Path
    bag_id: str
    heatmap_name: str
    num_points: int
    output_path: Path | None = None
    image_output_path: Path | None = None
    smoothed_image_output_path: Path | None = None
    top_tiles_output_path: Path | None = None


@dataclass(frozen=True)
class _ArtifactRenderContext:
    """Rendering context derived from the slide artifact.

    Attributes:
        source_coords_xy: Level-0 top-left coordinates shaped ``(N, 2)``.
        source_tile_sizes: Read-window width/height shaped ``(N, 2)``.
        source_read_levels: Pyramid level shaped ``(N,)`` aligned with
            ``source_coords_xy``.
        tiling_spec: Persisted bag tiling specification.
        slide_overview_jpeg: Optional JPEG overview bytes for the bag.
        source_slide_path: Optional source WSI path used for full-resolution
            tile previews.
    """

    source_coords_xy: np.ndarray
    source_tile_sizes: np.ndarray
    source_read_levels: np.ndarray
    tiling_spec: dict[str, Any]
    slide_overview_jpeg: bytes | None = None
    source_slide_path: Path | None = None


def create_inference_heatmap(
    *,
    artifact_path: str | Path,
    bag_id: str,
    scores_path: str | Path,
    heatmap_backend: str = "torchmil",
    heatmap_name: str = "torchmil",
    output_path: str | Path | None = None,
    image_output_path: str | Path | None = None,
    coords_path: str | Path | None = None,
    mask_path: str | Path | None = None,
    model_path: str | Path | None = None,
    slide_path: str | Path | None = None,
    colormap: str = "inferno",
    tile_alpha: float = 0.65,
    smoothed_alpha: float = 0.8,
    smoothing_sigma_scale: float = 0.75,
    top_k_tiles: int = 10,
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
        image_output_path: Optional PNG path containing a tile-aligned heatmap
            preview for quick visual inspection.
        coords_path: Optional `.npy`, `.npz`, or JSON coordinate file shaped
            `[N, 2]`. If omitted, coords are read from the H5 bag coords and the
            first two columns are used.
        mask_path: Optional `.npy`, `.npz`, or JSON boolean/binary mask shaped
            `[N]`.
        model_path: Optional model checkpoint path recorded in heatmap metadata.
        slide_path: Optional source WSI path used to render full-resolution
            top-tile previews. When omitted, the renderer falls back to
            artifact metadata when available.
        colormap: Matplotlib colormap name for exact and smoothed heatmaps.
        tile_alpha: Alpha applied to exact square-tile overlays in ``[0, 1]``.
        smoothed_alpha: Maximum alpha applied to the smoothed cloud overlay in
            ``[0, 1]``.
        smoothing_sigma_scale: Gaussian blur scale relative to the mean tile
            size in thumbnail pixels.
        top_k_tiles: Number of highest-scoring tiles to summarize visually.

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
    scores = _load_array(scores_path, expected_ndim=1, name="scores").astype(
        np.float32, copy=False
    )
    artifact_render_context = _read_artifact_render_context(artifact, bag_id)
    coords = (
        _load_array(coords_path, expected_ndim=2, name="coords")
        if coords_path is not None
        else artifact_render_context.source_coords_xy
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
        "colormap": colormap,
        "tile_alpha": float(tile_alpha),
        "smoothed_alpha": float(smoothed_alpha),
        "smoothing_sigma_scale": float(smoothing_sigma_scale),
        "top_k_tiles": int(top_k_tiles),
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
        _write_json_sidecar(
            out_path, coords=heatmap_coords, scores=heatmap_scores, metadata=metadata
        )
    image_out_path = Path(image_output_path) if image_output_path is not None else None
    smoothed_image_out_path: Path | None = None
    top_tiles_out_path: Path | None = None
    if image_out_path is not None:
        smoothed_image_out_path = _default_smoothed_heatmap_path(image_out_path)
        top_tiles_out_path = _default_top_tiles_path(image_out_path)
        tile_sizes = _match_tile_sizes_to_heatmap_coords(
            coords=heatmap_coords,
            source_coords=artifact_render_context.source_coords_xy,
            source_tile_sizes=artifact_render_context.source_tile_sizes,
        )
        read_levels = _match_tile_read_levels_to_heatmap_coords(
            coords=heatmap_coords,
            source_coords=artifact_render_context.source_coords_xy,
            source_read_levels=artifact_render_context.source_read_levels,
        )
        _write_heatmap_png(
            output_path=image_out_path,
            coords=heatmap_coords,
            scores=heatmap_scores,
            tile_sizes=tile_sizes,
            read_levels=read_levels,
            slide_overview_jpeg=artifact_render_context.slide_overview_jpeg,
            tiling_spec=artifact_render_context.tiling_spec,
            slide_path=(
                Path(slide_path)
                if slide_path is not None
                else artifact_render_context.source_slide_path
            ),
            smoothed_output_path=smoothed_image_out_path,
            top_tiles_output_path=top_tiles_out_path,
            colormap=colormap,
            tile_alpha=tile_alpha,
            smoothed_alpha=smoothed_alpha,
            smoothing_sigma_scale=smoothing_sigma_scale,
            top_k_tiles=top_k_tiles,
        )

    return InferenceHeatmapResult(
        artifact_path=artifact,
        bag_id=bag_id,
        heatmap_name=heatmap_name,
        num_points=int(heatmap_scores.shape[0]),
        output_path=out_path,
        image_output_path=image_out_path,
        smoothed_image_output_path=smoothed_image_out_path,
        top_tiles_output_path=top_tiles_out_path,
    )


def _resolve_heatmap_explainer_key(heatmap_backend: str) -> str:
    """Map one user-facing backend selector to a registry explainer key."""

    if heatmap_backend == "torchmil":
        return "torchmil_heatmap"
    return heatmap_backend


def _read_artifact_render_context(
    artifact_path: Path, bag_id: str
) -> _ArtifactRenderContext:
    """Read rendering context from an H5 artifact.

    Args:
        artifact_path: Slide artifact path.
        bag_id: Bag identifier resolved inside the slide artifact.

    Returns:
        _ArtifactRenderContext: Coordinates, tile sizes, tiling metadata, and
        optional overview bytes used to render previews.
    """

    try:
        with FileHandleH5(artifact_path, mode="r") as slide_artifact:
            coords = tiles_io.read_coords(slide_artifact, bag_id)
            try:
                tiling_spec = tiles_io.read_tiling_spec(slide_artifact, bag_id)
            except Exception:
                tiling_spec = {}
            overview_key = DEFAULT_LAYOUT.tiles_overview_dataset(bag_id)
            overview = (
                bytes(slide_artifact.h5[overview_key][()])
                if overview_key in slide_artifact.h5
                else None
            )
        return _ArtifactRenderContext(
            source_coords_xy=coords[:, :2].astype(np.float32, copy=False),
            source_tile_sizes=coords[:, 2:4].astype(np.float32, copy=False),
            source_read_levels=coords[:, 4].astype(np.int32, copy=False),
            tiling_spec=tiling_spec,
            slide_overview_jpeg=overview,
            source_slide_path=_resolve_source_slide_path(tiling_spec),
        )
    except Exception:
        empty = np.zeros((0, 2), dtype=np.float32)
        return _ArtifactRenderContext(
            source_coords_xy=empty,
            source_tile_sizes=empty.copy(),
            source_read_levels=np.zeros((0,), dtype=np.int32),
            tiling_spec={},
            slide_overview_jpeg=None,
            source_slide_path=None,
        )


def _resolve_source_slide_path(tiling_spec: dict[str, Any]) -> Path | None:
    """Resolve an optional persisted source WSI path from tiling metadata."""

    raw_path = tiling_spec.get("source_slide_path")
    if raw_path is None:
        return None
    try:
        path = Path(str(raw_path)).expanduser().resolve()
    except Exception:
        return None
    return path if path.exists() else None


def _load_array(
    path: str | Path | None, *, expected_ndim: int, name: str
) -> np.ndarray:
    """Load one numeric array from disk and validate rank/finite values.

    Args:
        path: Input file path. Supported formats are ``.npy``, ``.npz``, and
            JSON lists.
        expected_ndim: Required array rank.
        name: Human-readable field name used in validation errors.

    Returns:
        np.ndarray: Loaded array with exactly ``expected_ndim`` dimensions.
    """

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
        raise ValueError(
            f"{name} must have rank {expected_ndim}. Got shape {arr.shape}."
        )
    if not np.isfinite(arr.astype(np.float32, copy=False)).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return arr


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    """Detach one tensor and move it to CPU NumPy storage."""

    return value.detach().cpu().numpy()


def _write_json_sidecar(
    output_path: Path,
    *,
    coords: np.ndarray,
    scores: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Write one JSON heatmap sidecar for non-HDF5 consumers.

    Args:
        output_path: Output JSON path.
        coords: Coordinate array shaped ``(N, 2)``.
        scores: Score array shaped ``(N,)``.
        metadata: JSON-serializable metadata dictionary.
    """

    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "coords": coords.tolist(),
        "scores": scores.tolist(),
        "metadata": metadata,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _write_heatmap_png(
    output_path: Path,
    *,
    coords: np.ndarray,
    scores: np.ndarray,
    tile_sizes: np.ndarray,
    read_levels: np.ndarray,
    slide_overview_jpeg: bytes | None = None,
    tiling_spec: dict[str, Any] | None = None,
    slide_path: Path | None = None,
    smoothed_output_path: Path | None = None,
    top_tiles_output_path: Path | None = None,
    colormap: str = "inferno",
    tile_alpha: float = 0.65,
    smoothed_alpha: float = 0.8,
    smoothing_sigma_scale: float = 0.75,
    top_k_tiles: int = 10,
) -> None:
    """Render exact-tile and smoothed heatmap previews.

    Args:
        output_path: PNG path for the exact square-tile heatmap.
        coords: Level-0 top-left coordinates shaped ``(N, 2)``.
        scores: Normalized heatmap scores shaped ``(N,)`` with values in
            ``[0, 1]``.
        tile_sizes: Read-window width/height shaped ``(N, 2)``.
        read_levels: Pyramid levels shaped ``(N,)`` aligned with ``coords``.
        slide_overview_jpeg: Optional overview JPEG bytes previously rendered in
            thumbnail pixel space.
        tiling_spec: Optional bag tiling metadata. When overview scaling
            metadata is present, it is used to align the heatmap exactly to the
            overview image.
        slide_path: Optional source WSI path used to render full-resolution
            tile previews for the ranked gallery.
        smoothed_output_path: Optional PNG path for a blurred cloud rendering.
        top_tiles_output_path: Optional PNG path for a ranked top-K tiles table.
        colormap: Matplotlib colormap name.
        tile_alpha: Alpha for the exact tile overlay.
        smoothed_alpha: Maximum alpha for the smoothed overlay.
        smoothing_sigma_scale: Blur sigma multiplier relative to tile size.
        top_k_tiles: Number of highest-scoring tiles to summarize.
    """

    plt = _load_matplotlib_pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if smoothed_output_path is not None:
        smoothed_output_path.parent.mkdir(parents=True, exist_ok=True)
    if top_tiles_output_path is not None:
        top_tiles_output_path.parent.mkdir(parents=True, exist_ok=True)

    if coords.shape != tile_sizes.shape:
        raise ValueError(
            "tile_sizes must align row-wise with coords. "
            f"Got coords {coords.shape} and tile_sizes {tile_sizes.shape}."
        )
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (N,2). Got {coords.shape}.")
    if tile_sizes.ndim != 2 or tile_sizes.shape[1] != 2:
        raise ValueError(f"tile_sizes must have shape (N,2). Got {tile_sizes.shape}.")
    if scores.ndim != 1 or scores.shape[0] != coords.shape[0]:
        raise ValueError(
            "scores must have shape (N,) aligned with coords. "
            f"Got scores {scores.shape} and coords {coords.shape}."
        )
    if read_levels.ndim != 1 or read_levels.shape[0] != coords.shape[0]:
        raise ValueError(
            "read_levels must have shape (N,) aligned with coords. "
            f"Got read_levels {read_levels.shape} and coords {coords.shape}."
        )

    overview_arr = _decode_overview_image(slide_overview_jpeg)
    render_info = _build_render_space(
        coords=coords,
        tile_sizes=tile_sizes,
        overview_arr=overview_arr,
        tiling_spec=tiling_spec,
    )
    tile_rectangles = _coords_to_pixel_rectangles(
        coords=coords,
        tile_sizes=tile_sizes,
        downscale_x=render_info["downscale_x"],
        downscale_y=render_info["downscale_y"],
    )
    exact_fig = _plot_exact_tile_heatmap(
        plt=plt,
        background=overview_arr,
        tile_rectangles=tile_rectangles,
        scores=scores,
        title="Inference Heatmap",
        colormap=colormap,
        tile_alpha=tile_alpha,
    )
    exact_fig.savefig(output_path, dpi=150)
    plt.close(exact_fig)

    if smoothed_output_path is not None:
        smoothed_fig = _plot_smoothed_heatmap(
            plt=plt,
            background=overview_arr,
            tile_rectangles=tile_rectangles,
            scores=scores,
            canvas_shape=render_info["canvas_shape"],
            title="Inference Heatmap (Smoothed)",
            colormap=colormap,
            smoothed_alpha=smoothed_alpha,
            smoothing_sigma_scale=smoothing_sigma_scale,
        )
        smoothed_fig.savefig(smoothed_output_path, dpi=150)
        plt.close(smoothed_fig)
    if top_tiles_output_path is not None:
        top_tile_images = _read_top_tile_images(
            slide_path=slide_path,
            coords=coords,
            tile_sizes=tile_sizes,
            read_levels=read_levels,
            tiling_spec=tiling_spec,
            scores=scores,
        )
        top_tiles_fig = _plot_top_k_tiles_table(
            plt=plt,
            background=overview_arr,
            tile_rectangles=tile_rectangles,
            scores=scores,
            coords=coords,
            tile_images=top_tile_images,
            top_k_tiles=top_k_tiles,
            title="Top Tiles",
        )
        top_tiles_fig.savefig(top_tiles_output_path, dpi=150)
        plt.close(top_tiles_fig)


def _default_smoothed_heatmap_path(image_output_path: Path) -> Path:
    """Return the default sibling PNG path for the smoothed heatmap preview."""

    return image_output_path.with_name(
        f"{image_output_path.stem}_smoothed{image_output_path.suffix}"
    )


def _default_top_tiles_path(image_output_path: Path) -> Path:
    """Return the default sibling PNG path for the top-K tile summary."""

    return image_output_path.with_name(
        f"{image_output_path.stem}_top_tiles{image_output_path.suffix}"
    )


def _decode_overview_image(slide_overview_jpeg: bytes | None) -> np.ndarray | None:
    """Decode overview bytes into an RGB image array when available."""

    if slide_overview_jpeg is None:
        return None
    try:
        import io

        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(slide_overview_jpeg)).convert("RGB"))
    except Exception:
        return None


def _build_render_space(
    *,
    coords: np.ndarray,
    tile_sizes: np.ndarray,
    overview_arr: np.ndarray | None,
    tiling_spec: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve the thumbnail pixel-space mapping used for rendering.

    When overview metadata is available, the mapping matches the stored
    tiles-overview image exactly. Older artifacts fall back to a deterministic
    estimate based on the observed tile coverage.
    """

    max_end_x = float(np.max(coords[:, 0] + tile_sizes[:, 0]))
    max_end_y = float(np.max(coords[:, 1] + tile_sizes[:, 1]))
    if overview_arr is not None:
        canvas_height, canvas_width = (
            int(overview_arr.shape[0]),
            int(overview_arr.shape[1]),
        )
    else:
        # Keep the fallback canvas compact while preserving aspect ratio.
        longest_side = max(max_end_x, max_end_y, 1.0)
        scale = 1200.0 / longest_side
        canvas_width = max(1, int(np.ceil(max_end_x * scale)))
        canvas_height = max(1, int(np.ceil(max_end_y * scale)))

    downscale_x = _extract_positive_float(tiling_spec, "tiles_overview_downscale_x")
    downscale_y = _extract_positive_float(tiling_spec, "tiles_overview_downscale_y")
    if downscale_x is None or downscale_y is None:
        downscale_x = max(max_end_x / max(float(canvas_width), 1.0), 1.0)
        downscale_y = max(max_end_y / max(float(canvas_height), 1.0), 1.0)

    return {
        "canvas_shape": (canvas_height, canvas_width),
        "downscale_x": float(downscale_x),
        "downscale_y": float(downscale_y),
    }


def _extract_positive_float(metadata: dict[str, Any] | None, key: str) -> float | None:
    """Read a positive float from optional metadata."""

    if metadata is None or key not in metadata:
        return None
    try:
        value = float(metadata[key])
    except Exception:
        return None
    return value if value > 0 else None


def _match_tile_sizes_to_heatmap_coords(
    *,
    coords: np.ndarray,
    source_coords: np.ndarray,
    source_tile_sizes: np.ndarray,
) -> np.ndarray:
    """Map persisted heatmap coordinates back to their source tile footprints.

    Args:
        coords: Heatmap coordinates shaped ``(N, 2)``.
        source_coords: Source bag coordinates shaped ``(M, 2)``.
        source_tile_sizes: Source tile sizes shaped ``(M, 2)``.

    Returns:
        np.ndarray: Tile sizes shaped ``(N, 2)`` aligned with ``coords``.
    """

    if source_coords.shape != source_tile_sizes.shape:
        raise ValueError(
            "source_coords and source_tile_sizes must share shape (M,2). "
            f"Got {source_coords.shape} and {source_tile_sizes.shape}."
        )

    coord_to_sizes: dict[tuple[float, float], list[np.ndarray]] = {}
    for idx in range(source_coords.shape[0]):
        key = (float(source_coords[idx, 0]), float(source_coords[idx, 1]))
        coord_to_sizes.setdefault(key, []).append(source_tile_sizes[idx])

    matched_sizes: list[np.ndarray] = []
    fallback_size = _infer_default_tile_size(source_tile_sizes)
    for idx in range(coords.shape[0]):
        key = (float(coords[idx, 0]), float(coords[idx, 1]))
        queue = coord_to_sizes.get(key)
        matched_sizes.append(
            np.asarray(queue.pop(0) if queue else fallback_size, dtype=np.float32)
        )
    return np.vstack(matched_sizes).astype(np.float32, copy=False)


def _match_tile_read_levels_to_heatmap_coords(
    *,
    coords: np.ndarray,
    source_coords: np.ndarray,
    source_read_levels: np.ndarray,
) -> np.ndarray:
    """Map persisted heatmap coordinates back to their source read levels."""

    if source_coords.ndim != 2 or source_coords.shape[1] != 2:
        raise ValueError(
            "source_coords must have shape (M,2). "
            f"Got {source_coords.shape}."
        )
    if source_read_levels.ndim != 1 or source_read_levels.shape[0] != source_coords.shape[0]:
        raise ValueError(
            "source_read_levels must have shape (M,) aligned with source_coords. "
            f"Got {source_read_levels.shape} and {source_coords.shape}."
        )

    coord_to_levels: dict[tuple[float, float], list[int]] = {}
    for idx in range(source_coords.shape[0]):
        key = (float(source_coords[idx, 0]), float(source_coords[idx, 1]))
        coord_to_levels.setdefault(key, []).append(int(source_read_levels[idx]))

    matched_levels: list[int] = []
    for idx in range(coords.shape[0]):
        key = (float(coords[idx, 0]), float(coords[idx, 1]))
        queue = coord_to_levels.get(key)
        matched_levels.append(int(queue.pop(0)) if queue else 0)
    return np.asarray(matched_levels, dtype=np.int32)


def _infer_default_tile_size(source_tile_sizes: np.ndarray) -> np.ndarray:
    """Infer a positive fallback tile size from source geometry."""

    if source_tile_sizes.size == 0:
        return np.asarray([1.0, 1.0], dtype=np.float32)
    positive = np.where(source_tile_sizes > 0, source_tile_sizes, np.nan)
    medians = np.nanmedian(positive, axis=0)
    if not np.isfinite(medians).all():
        return np.asarray([1.0, 1.0], dtype=np.float32)
    return np.maximum(medians.astype(np.float32, copy=False), 1.0)


def _coords_to_pixel_rectangles(
    *,
    coords: np.ndarray,
    tile_sizes: np.ndarray,
    downscale_x: float,
    downscale_y: float,
) -> np.ndarray:
    """Project level-0 tile rectangles into thumbnail pixel space."""

    x0 = np.round(coords[:, 0] / downscale_x).astype(np.int32, copy=False)
    y0 = np.round(coords[:, 1] / downscale_y).astype(np.int32, copy=False)
    widths = np.maximum(
        np.round(tile_sizes[:, 0] / downscale_x).astype(np.int32, copy=False), 1
    )
    heights = np.maximum(
        np.round(tile_sizes[:, 1] / downscale_y).astype(np.int32, copy=False), 1
    )
    return np.stack([x0, y0, widths, heights], axis=1)


def _plot_exact_tile_heatmap(
    *,
    plt: Any,
    background: np.ndarray | None,
    tile_rectangles: np.ndarray,
    scores: np.ndarray,
    title: str,
    colormap: str,
    tile_alpha: float,
) -> Any:
    """Plot the exact square-tile heatmap in thumbnail pixel space."""

    from matplotlib import cm
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(8, 6))
    if background is not None:
        ax.imshow(background, origin="upper", zorder=0)

    patches = [
        Rectangle((float(x0), float(y0)), float(width), float(height))
        for x0, y0, width, height in tile_rectangles.tolist()
    ]
    norm = Normalize(vmin=0.0, vmax=1.0)
    collection = PatchCollection(
        patches,
        cmap=colormap,
        norm=norm,
        alpha=tile_alpha,
        linewidths=0.0,
        edgecolors="none",
        zorder=1,
    )
    collection.set_array(scores.astype(np.float32, copy=False))
    ax.add_collection(collection)
    _style_thumbnail_axis(
        ax=ax, background=background, tile_rectangles=tile_rectangles, title=title
    )
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=colormap),
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Score",
    )
    fig.tight_layout()
    return fig


def _plot_smoothed_heatmap(
    *,
    plt: Any,
    background: np.ndarray | None,
    tile_rectangles: np.ndarray,
    scores: np.ndarray,
    canvas_shape: tuple[int, int],
    title: str,
    colormap: str,
    smoothed_alpha: float,
    smoothing_sigma_scale: float,
) -> Any:
    """Plot a blurred cloud rendering of the tile heatmap."""

    from matplotlib import cm
    from matplotlib.colors import Normalize

    value_grid, weight_grid = _rasterize_tile_scores(
        tile_rectangles=tile_rectangles,
        scores=scores,
        canvas_shape=canvas_shape,
    )
    tile_size_pixels = np.median(
        tile_rectangles[:, 2:4].astype(np.float32, copy=False), axis=0
    )
    sigma = max(float(np.mean(tile_size_pixels)) * float(smoothing_sigma_scale), 1.0)
    blurred_values = _gaussian_blur_2d(value_grid, sigma=sigma)
    blurred_weights = _gaussian_blur_2d(weight_grid, sigma=sigma)
    smoothed = np.divide(
        blurred_values,
        np.maximum(blurred_weights, 1e-6),
        out=np.zeros_like(blurred_values, dtype=np.float32),
        where=blurred_weights > 1e-6,
    )
    alpha = np.clip(
        blurred_weights / np.maximum(np.max(blurred_weights), 1e-6), 0.0, 1.0
    ) * float(smoothed_alpha)

    fig, ax = plt.subplots(figsize=(8, 6))
    if background is not None:
        ax.imshow(background, origin="upper", zorder=0)
    norm = Normalize(vmin=0.0, vmax=1.0)
    ax.imshow(
        smoothed,
        origin="upper",
        cmap=colormap,
        norm=norm,
        alpha=alpha,
        interpolation="bilinear",
        zorder=1,
    )
    _style_thumbnail_axis(
        ax=ax, background=background, tile_rectangles=tile_rectangles, title=title
    )
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=colormap),
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Score",
    )
    fig.tight_layout()
    return fig


def _plot_top_k_tiles_table(
    *,
    plt: Any,
    background: np.ndarray | None,
    tile_rectangles: np.ndarray,
    scores: np.ndarray,
    coords: np.ndarray,
    tile_images: list[np.ndarray] | None,
    top_k_tiles: int,
    title: str,
) -> Any:
    """Render a top-K tiles gallery with full-resolution previews when available."""

    num_tiles = max(1, min(int(top_k_tiles), int(scores.shape[0])))
    ranking = np.argsort(scores)[::-1][:num_tiles]
    num_rows, num_cols = _top_tiles_grid_shape(num_tiles)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(max(2.8 * num_cols, 3.2), max(3.4 * num_rows, 3.6)),
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for gallery_index, item_index in enumerate(ranking):
        tile_ax = axes_flat[gallery_index]
        tile_ax.axis("off")
        tile_ax.imshow(
            (
                tile_images[gallery_index]
                if tile_images is not None
                else _crop_tile_preview(
                    background=background,
                    rectangle=tile_rectangles[item_index],
                )
            ),
            origin="upper",
        )
        tile_ax.set_title(
            f"#{gallery_index + 1} score={float(scores[item_index]):.4f}",
            fontsize=9,
        )
        tile_ax.text(
            0.5,
            -0.08,
            f"x={int(coords[item_index, 0])}, y={int(coords[item_index, 1])}",
            fontsize=8,
            ha="center",
            va="top",
            transform=tile_ax.transAxes,
        )
    for empty_ax in axes_flat[num_tiles:]:
        empty_ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _top_tiles_grid_shape(num_tiles: int) -> tuple[int, int]:
    """Return a compact gallery shape for the ranked top-tile montage."""

    if num_tiles <= 0:
        raise ValueError(f"num_tiles must be positive. Got {num_tiles}.")
    num_cols = max(1, int(np.ceil(np.sqrt(float(num_tiles)))))
    num_rows = int(np.ceil(float(num_tiles) / float(num_cols)))
    return num_rows, num_cols


def _read_top_tile_images(
    *,
    slide_path: Path | None,
    coords: np.ndarray,
    tile_sizes: np.ndarray,
    read_levels: np.ndarray,
    tiling_spec: dict[str, Any] | None,
    scores: np.ndarray,
) -> list[np.ndarray] | None:
    """Read top-tile previews from the source WSI when slide context exists."""

    if slide_path is None or not slide_path.exists():
        return None

    ranking = np.argsort(scores)[::-1]
    target_tile_size = _extract_display_tile_size(tiling_spec)

    try:
        from wsidata import open_wsi
    except Exception:
        logger.warning(
            "Skipping full-resolution top-tile previews because wsidata is unavailable."
        )
        return None

    try:
        wsi = open_wsi(slide_path)
    except Exception:
        logger.warning("Failed to open source slide for top-tile previews: %s", slide_path)
        return None

    previews: list[np.ndarray] = []
    try:
        for item_index in ranking:
            patch = wsi.read_region(
                int(round(float(coords[item_index, 0]))),
                int(round(float(coords[item_index, 1]))),
                max(int(round(float(tile_sizes[item_index, 0]))), 1),
                max(int(round(float(tile_sizes[item_index, 1]))), 1),
                level=max(int(read_levels[item_index]), 0),
            )
            previews.append(
                _resize_tile_preview(
                    _coerce_rgb_uint8_array(patch),
                    target_size_px=target_tile_size,
                )
            )
    except Exception:
        logger.warning(
            "Falling back to thumbnail crops because source tile reads failed for %s.",
            slide_path,
            exc_info=True,
        )
        return None
    finally:
        wsi.close()

    return previews


def _extract_display_tile_size(tiling_spec: dict[str, Any] | None) -> int:
    """Return the rendered tile size used in the top-tile gallery."""

    if tiling_spec is None:
        return 256
    try:
        tile_px = int(tiling_spec.get("tile_px", 256))
    except Exception:
        return 256
    return max(tile_px, 1)


def _coerce_rgb_uint8_array(image: Any) -> np.ndarray:
    """Convert region-read output into an RGB ``uint8`` image array."""

    from PIL import Image

    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Unsupported tile preview shape: {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _resize_tile_preview(image: np.ndarray, *, target_size_px: int) -> np.ndarray:
    """Resize a tile preview to a consistent square display size."""

    from PIL import Image

    pil_image = Image.fromarray(image, mode="RGB")
    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR  # type: ignore[attr-defined]
    return np.asarray(
        pil_image.resize((int(target_size_px), int(target_size_px)), resample=resample),
        dtype=np.uint8,
    )


def _crop_tile_preview(
    *,
    background: np.ndarray | None,
    rectangle: np.ndarray,
) -> np.ndarray:
    """Return a tile preview crop for the top-K tiles table."""

    if background is None:
        height = max(int(rectangle[3]), 1)
        width = max(int(rectangle[2]), 1)
        return np.full((height, width, 3), 235, dtype=np.uint8)
    x0, y0, width, height = [int(v) for v in rectangle]
    x1 = min(background.shape[1], x0 + max(width, 1))
    y1 = min(background.shape[0], y0 + max(height, 1))
    x0 = max(0, x0)
    y0 = max(0, y0)
    if x1 <= x0 or y1 <= y0:
        return np.full((max(height, 1), max(width, 1), 3), 235, dtype=np.uint8)
    return background[y0:y1, x0:x1]


def _style_thumbnail_axis(
    *, ax: Any, background: np.ndarray | None, tile_rectangles: np.ndarray, title: str
) -> None:
    """Apply consistent styling to thumbnail-space heatmap plots."""

    if background is not None:
        ax.set_xlim(0, background.shape[1])
        ax.set_ylim(background.shape[0], 0)
    else:
        max_x = int(np.max(tile_rectangles[:, 0] + tile_rectangles[:, 2]))
        max_y = int(np.max(tile_rectangles[:, 1] + tile_rectangles[:, 3]))
        ax.set_xlim(0, max_x)
        ax.set_ylim(max_y, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("thumbnail x")
    ax.set_ylabel("thumbnail y")


def _rasterize_tile_scores(
    *,
    tile_rectangles: np.ndarray,
    scores: np.ndarray,
    canvas_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize tile scores into dense value/weight grids."""

    canvas_height, canvas_width = canvas_shape
    values = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    weights = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    for idx, rect in enumerate(tile_rectangles):
        x0, y0, width, height = [int(v) for v in rect]
        x1 = min(canvas_width, x0 + max(width, 1))
        y1 = min(canvas_height, y0 + max(height, 1))
        if x1 <= 0 or y1 <= 0 or x0 >= canvas_width or y0 >= canvas_height:
            continue
        x0_clip = max(0, x0)
        y0_clip = max(0, y0)
        values[y0_clip:y1, x0_clip:x1] += float(scores[idx])
        weights[y0_clip:y1, x0_clip:x1] += 1.0
    return values, weights


def _gaussian_blur_2d(array: np.ndarray, *, sigma: float) -> np.ndarray:
    """Apply a separable Gaussian blur without introducing SciPy."""

    if sigma <= 0:
        return array.astype(np.float32, copy=False)
    radius = max(int(np.ceil(3.0 * sigma)), 1)
    kernel_x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(kernel_x**2) / (2.0 * float(sigma) ** 2))
    kernel /= np.sum(kernel)
    blurred = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), axis=1, arr=array
    )
    blurred = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="same"), axis=0, arr=blurred
    )
    return blurred.astype(np.float32, copy=False)


def _load_matplotlib_pyplot():
    """Load matplotlib pyplot in headless ``Agg`` mode for artifact rendering."""

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt
