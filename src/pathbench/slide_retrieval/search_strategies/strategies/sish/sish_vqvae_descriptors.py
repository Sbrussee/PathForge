from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from pathbench.core.datasets.wsi_dataset import WSI
from pathbench.core.io.slide_artifacts.atomic import atomic_slide_artifact_write
from pathbench.core.io.slide_artifacts.base import FileHandleH5
from pathbench.core.io.slide_artifacts import tiles as tiles_io
from pathbench.core.io.slide_retrieval import descriptors as descriptors_io
from pathbench.slide_retrieval.representation_strategies.mean_rgb import (
    _build_slide_processor,
    _resolve_sample_slide_paths,
    _slide_retrieval_artifact_path,
)
from pathbench.slide_retrieval.search_strategies.strategies.sish.sish_vqvae import (
    LargeVectorQuantizedVAE_Encode,
)


SISH_VQVAE_DESCRIPTOR_NAME = "sish_vqvae_latent"


def resolve_sample_patch_sish_vqvae_latent(
    *,
    sample: Any,
    bag_id: str,
    config: Any,
    descriptor_name: str = SISH_VQVAE_DESCRIPTOR_NAME,
) -> np.ndarray:
    if sample is None:
        raise ValueError("sample is required to resolve SISH VQ-VAE descriptors.")
    if config is None:
        raise ValueError("config is required to resolve SISH VQ-VAE descriptors.")

    artifact_paths = [Path(path) for path in list(getattr(sample, "artifact_paths", []) or [])]
    slide_ids = [str(slide_id) for slide_id in list(getattr(sample, "slide_ids", []) or [])]

    if not artifact_paths:
        raise ValueError("sample.artifact_paths is required to resolve SISH VQ-VAE descriptors.")
    if len(slide_ids) != len(artifact_paths):
        raise ValueError(
            "sample.slide_ids and sample.artifact_paths must have the same length. "
            f"Got {len(slide_ids)} and {len(artifact_paths)}."
        )

    model = None
    slide_processor = None
    slide_paths_by_id: dict[str, Path | None] | None = None
    descriptor_parts: list[np.ndarray] = []
    for slide_id, artifact_path in zip(slide_ids, artifact_paths, strict=False):
        retrieval_artifact_path = _slide_retrieval_artifact_path(
            slide_artifact_path=artifact_path,
            slide_id=slide_id,
        )
        with FileHandleH5(artifact_path, mode="r") as slide_artifact:
            expected_rows = int(tiles_io.coords_num_rows(slide_artifact, bag_id=bag_id))

        if retrieval_artifact_path.is_file():
            with FileHandleH5(retrieval_artifact_path, mode="r") as retrieval_artifact:
                if descriptors_io.descriptor_exists(
                    retrieval_artifact,
                    tile_id=bag_id,
                    descriptor_name=descriptor_name,
                    expected_rows=expected_rows,
                ):
                    descriptor_parts.append(
                        descriptors_io.read_descriptor(
                            retrieval_artifact,
                            tile_id=bag_id,
                            descriptor_name=descriptor_name,
                        )
                    )
                    continue

        if slide_paths_by_id is None:
            slide_paths_by_id = _resolve_sample_slide_paths(sample=sample, config=config)
        if slide_processor is None:
            slide_processor = _build_slide_processor(config=config)
        if model is None:
            model = _load_sish_vqvae_encoder(config=config)

        descriptor_parts.append(
            _resolve_slide_patch_sish_vqvae_latent(
                slide_artifact_path=artifact_path,
                retrieval_artifact_path=retrieval_artifact_path,
                slide_path=slide_paths_by_id.get(slide_id),
                bag_id=bag_id,
                slide_processor=slide_processor,
                slide_id=slide_id,
                model=model,
                descriptor_name=descriptor_name,
                config=config,
            )
        )

    if not descriptor_parts:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(descriptor_parts, axis=0).astype(np.float32, copy=False)


def load_or_create_slide_patch_sish_vqvae_latent(
    *,
    slide_artifact: FileHandleH5,
    retrieval_artifact_path: Path,
    slide_path: Path | None,
    bag_id: str,
    slide_processor: Any,
    slide_id: str,
    model: torch.nn.Module,
    descriptor_name: str = SISH_VQVAE_DESCRIPTOR_NAME,
    batch_size: int = 8,
) -> np.ndarray:
    expected_rows = int(tiles_io.coords_num_rows(slide_artifact, bag_id=bag_id))
    if slide_path is None or not slide_path.is_file():
        raise FileNotFoundError(
            "Missing stored SISH VQ-VAE descriptors and no source slide is "
            f"available to compute them for slide '{slide_id}' at bag_id='{bag_id}'."
        )

    if retrieval_artifact_path.is_file():
        with FileHandleH5(retrieval_artifact_path, mode="r") as retrieval_artifact:
            if descriptors_io.descriptor_exists(
                retrieval_artifact,
                tile_id=bag_id,
                descriptor_name=descriptor_name,
                expected_rows=expected_rows,
            ):
                return descriptors_io.read_descriptor(
                    retrieval_artifact,
                    tile_id=bag_id,
                    descriptor_name=descriptor_name,
                )

    descriptor_matrix = _create_slide_patch_sish_vqvae_latent(
        slide_artifact=slide_artifact,
        slide_path=slide_path,
        bag_id=bag_id,
        slide_processor=slide_processor,
        slide_id=slide_id,
        model=model,
        batch_size=batch_size,
    )
    with atomic_slide_artifact_write(retrieval_artifact_path) as retrieval_artifact:
        descriptors_io.write_descriptor(
            retrieval_artifact,
            tile_id=bag_id,
            descriptor_name=descriptor_name,
            descriptor_matrix=descriptor_matrix,
        )
    return descriptor_matrix


def _resolve_slide_patch_sish_vqvae_latent(
    *,
    slide_artifact_path: Path,
    retrieval_artifact_path: Path,
    slide_path: Path | None,
    bag_id: str,
    slide_processor: Any,
    slide_id: str,
    model: torch.nn.Module,
    descriptor_name: str,
    config: Any,
) -> np.ndarray:
    with FileHandleH5(slide_artifact_path, mode="r") as slide_artifact:
        batch_size = int(
            _get_config_value(
                config,
                [
                    ("experiment", "sish", "vqvae_batch_size"),
                    ("experiment", "SISH_metrics", "vqvae_batch_size"),
                    ("sish", "vqvae_batch_size"),
                ],
                default=8,
            )
        )
        return load_or_create_slide_patch_sish_vqvae_latent(
            slide_artifact=slide_artifact,
            retrieval_artifact_path=retrieval_artifact_path,
            slide_path=slide_path,
            bag_id=bag_id,
            slide_processor=slide_processor,
            slide_id=slide_id,
            model=model,
            descriptor_name=descriptor_name,
            batch_size=batch_size,
        )


def _create_slide_patch_sish_vqvae_latent(
    *,
    slide_artifact: FileHandleH5,
    slide_path: Path,
    bag_id: str,
    slide_processor: Any,
    slide_id: str,
    model: torch.nn.Module,
    batch_size: int,
) -> np.ndarray:
    coords = np.asarray(tiles_io.read_coords(slide_artifact, bag_id=bag_id), dtype=np.int32)
    expected_rows = int(coords.shape[0])
    slide_wsi = WSI(
        slide=slide_id,
        patient="",
        category="",
        path=slide_path,
        artifact_path=slide_artifact.path,
    )
    slide_processor.load_wsi(slide_wsi)
    try:
        latent_rows: list[np.ndarray] = []
        batch_tensors: list[torch.Tensor] = []
        for row in coords:
            x0, y0, read_w, read_h, read_level = [int(value) for value in row]
            patch = slide_processor.read_patch_region(
                slide_wsi,
                x=x0,
                y=y0,
                width=read_w,
                height=read_h,
                level=read_level,
            )
            patch_array = np.asarray(patch, dtype=np.uint8)
            if patch_array.ndim != 3 or patch_array.shape[2] != 3:
                raise ValueError(
                    "Patch RGB reads must have shape (H,W,3). "
                    f"Got {patch_array.shape} for slide '{slide_id}'."
                )

            tensor = torch.from_numpy(patch_array).permute(2, 0, 1).float() / 255.0
            tensor = (2.0 * tensor) - 1.0
            batch_tensors.append(tensor)

            if len(batch_tensors) >= batch_size:
                latent_rows.extend(_encode_latent_batch(batch_tensors, model))
                batch_tensors = []

        if batch_tensors:
            latent_rows.extend(_encode_latent_batch(batch_tensors, model))
    finally:
        slide_processor.close_wsi(slide_wsi)

    if len(latent_rows) != expected_rows:
        raise RuntimeError(
            "Failed to materialize VQ-VAE latents for all patches. "
            f"Expected {expected_rows}, got {len(latent_rows)}."
        )

    descriptor_matrix = np.stack(latent_rows, axis=0).astype(np.float32, copy=False)
    if descriptor_matrix.ndim != 2 or descriptor_matrix.shape[0] != expected_rows:
        raise ValueError(
            "SISH VQ-VAE descriptor matrix must have shape (N, D). "
            f"Got {descriptor_matrix.shape}."
        )
    return descriptor_matrix


def _encode_latent_batch(
    batch_tensors: list[torch.Tensor],
    model: torch.nn.Module,
) -> list[np.ndarray]:
    device = next(model.parameters()).device
    with torch.no_grad():
        batch = torch.stack(batch_tensors, dim=0).to(device)
        latents = model(batch).detach().cpu().numpy()
    latents = np.asarray(latents)
    return [latent.reshape(-1).astype(np.float32, copy=False) for latent in latents]


def _load_sish_vqvae_encoder(*, config: Any) -> torch.nn.Module:
    checkpoint_path = _resolve_path(
        config,
        [
            ("experiment", "sish", "vqvae_checkpoint"),
            ("experiment", "SISH_metrics", "vqvae_checkpoint"),
            ("sish", "vqvae_checkpoint"),
        ],
    )
    if checkpoint_path is None:
        raise ValueError(
            "SISH descriptor creation requires config path 'sish.vqvae_checkpoint'."
        )

    model = LargeVectorQuantizedVAE_Encode(code_dim=256, code_size=128)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")["model"]
    encoder_weights = {
        key[len("module."):]: value
        for key, value in checkpoint.items()
        if key.startswith("module.encoder.") or key.startswith("module.codebook.")
    }
    model.load_state_dict(encoder_weights, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model


def _resolve_path(source: Any, candidate_paths: list[tuple[str, ...]]) -> Path | None:
    value = _get_config_value(source, candidate_paths, default=None)
    if value is None:
        return None
    return Path(value)


def _get_config_value(
    source: Any,
    candidate_paths: list[tuple[str, ...]],
    *,
    default: Any,
) -> Any:
    for path in candidate_paths:
        current = source
        for key in path:
            if current is None:
                break
            if isinstance(current, dict):
                current = current.get(key)
            else:
                current = getattr(current, key, None)
        if current is not None:
            return current
    return default
