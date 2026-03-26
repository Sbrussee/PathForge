from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pathbench.core.datasets.wsi_dataset import WSI
from pathbench.core.io.h5 import tiles as tiles_io
from pathbench.core.io.h5.base import FileHandleH5
from pathbench.slide_retrieval.mean_rgb import (
    _build_slide_processor,
    _resolve_sample_slide_paths,
)
from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathbench.slide_retrieval.search_strategies.strategies.sish.sish_vqvae import (
    LargeVectorQuantizedVAE_Encode,
)


def _scale_to_minus1_to_1(x: torch.Tensor) -> torch.Tensor:
    """Map input tensors from ``[0, 1]`` to ``[-1, 1]``."""
    return (2.0 * x) - 1.0


def _to_latent_semantic(
    latent: np.ndarray,
    codebook_semantic: dict[int, int],
) -> np.ndarray:
    """Map raw VQ-VAE latent codes to the semantic codebook indices."""
    latent_semantic = np.zeros_like(latent)
    for row_index in range(latent.shape[0]):
        for col_index in range(latent.shape[1]):
            latent_semantic[row_index, col_index] = codebook_semantic[
                int(latent[row_index, col_index])
            ]
    return latent_semantic


def _slide_to_index(
    latent: np.ndarray,
    codebook_semantic: dict[int, int],
    pool_layers: list[torch.nn.Module],
) -> np.ndarray:
    """Convert a batch of VQ-VAE latent maps into SISH integer indices."""
    latent_array = np.asarray(latent)
    if latent_array.ndim == 2:
        semantic = _to_latent_semantic(latent_array, codebook_semantic)
        feat = torch.from_numpy(semantic[np.newaxis, ...])
    elif latent_array.ndim == 3:
        semantic_batch = [
            _to_latent_semantic(latent_array[index], codebook_semantic)
            for index in range(latent_array.shape[0])
        ]
        feat = torch.from_numpy(np.stack(semantic_batch, axis=0))
    else:
        raise ValueError(
            "Expected VQ-VAE latent array with ndim 2 or 3. "
            f"Got {latent_array.ndim}."
        )

    level_sum_dict: dict[int, np.ndarray] = {}
    num_levels = list(range(len(pool_layers) + 1))
    for level in num_levels:
        if level == 0:
            level_sum_dict[level] = torch.sum(feat, (1, 2)).numpy().astype(float)
        else:
            feat = pool_layers[level - 1](feat)
            level_sum_dict[level] = torch.sum(feat, (1, 2)).numpy().astype(float)

    level_power = [0, 0, 1e6, 1e11]
    index = 0
    for level, power in enumerate(level_power):
        if level == 1:
            index = level_sum_dict[level].copy()
        elif level > 1:
            index += level_sum_dict[level] * power

    return np.asarray(index, dtype=np.int64)


@dataclass(slots=True)
class _SelectedPatchSpec:
    output_position: int
    slide_id: str
    artifact_path: Path
    slide_path: Path
    x: int
    y: int
    read_w: int
    read_h: int
    level: int


class SISHPrecompute:
    """
    Compute and persist per-patch SISH indices for retrieval representations.

    Semantic goal:
        Materialize the image-derived SISH integer indices once during
        retrieval-representation creation, then store them in
        ``representation.additional_data`` so the search strategy no longer
        needs the removed mosaic/TFRecord pipeline.
    """

    def __init__(self, *, config: Any) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._codebook_semantic: dict[int, int] | None = None
        self._vqvae: torch.nn.Module | None = None
        self._slide_processor = None
        self._pool_layers = [
            torch.nn.AvgPool2d((2, 2)),
            torch.nn.AvgPool2d((2, 2)),
            torch.nn.AvgPool2d((2, 2)),
        ]

    def enrich_representation(
        self,
        *,
        representation: RetrievalRepresentation,
        sample: Any,
        bag_id: str,
    ) -> RetrievalRepresentation:
        """
        Enrich one retrieval representation with SISH-specific additional data.

        Inputs:
            representation:
                Multi-vector patch representation with ``data.shape == (K, D)``.
            sample:
                Sample carrying ``slide_ids`` and ``artifact_paths``.
            bag_id:
                Canonical H5 bag id whose coords define the selected patch rows.

        Output:
            RetrievalRepresentation:
                Same object with added:
                - ``additional_data["sish_patch_indices"]`` shape ``(K,)``
                - ``additional_data["sish_packed_bits"]`` shape ``(K, ceil(D/8))``
        """
        if "sish_patch_indices" in representation.additional_data:
            if "sish_packed_bits" not in representation.additional_data:
                representation.additional_data["sish_packed_bits"] = self._pack_bits(
                    representation.data
                )
            return representation

        features = np.asarray(representation.data, dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        if features.ndim != 2:
            raise ValueError(
                "SISH preprocessing expects representation data with shape (N, D). "
                f"Got {features.shape}."
            )

        full_coords = self._load_sample_full_coords(sample=sample, bag_id=bag_id)
        selected_indices = self._resolve_selected_indices(
            representation=representation,
            full_coords=full_coords,
        )
        selected_coords = full_coords[selected_indices]
        self._validate_selected_coords(
            representation=representation,
            selected_full_coords=selected_coords[:, :2],
        )

        patch_specs = self._build_patch_specs(
            sample=sample,
            bag_id=bag_id,
            full_coords=full_coords,
            selected_indices=selected_indices,
        )
        patch_indices = self._encode_selected_patch_specs(patch_specs=patch_specs)

        representation.additional_data["sish_patch_indices"] = patch_indices.astype(
            np.int64,
            copy=False,
        )
        representation.additional_data["sish_packed_bits"] = self._pack_bits(features)
        return representation

    def _load_models(self) -> None:
        """Lazy-load the VQ-VAE encoder and semantic codebook from config."""
        if self._vqvae is not None and self._codebook_semantic is not None:
            return

        codebook_path = self._resolve_path(
            [
                ("experiment", "sish", "codebook_semantic"),
                ("experiment", "SISH_metrics", "codebook_semantic"),
                ("sish", "codebook_semantic"),
            ]
        )
        checkpoint_path = self._resolve_path(
            [
                ("experiment", "sish", "vqvae_checkpoint"),
                ("experiment", "SISH_metrics", "vqvae_checkpoint"),
                ("sish", "vqvae_checkpoint"),
            ]
        )
        if codebook_path is None or checkpoint_path is None:
            raise ValueError(
                "SISH preprocessing requires config paths for "
                "'codebook_semantic' and 'vqvae_checkpoint'."
            )

        self._codebook_semantic = torch.load(codebook_path, map_location="cpu")
        self._vqvae = LargeVectorQuantizedVAE_Encode(code_dim=256, code_size=128)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")["model"]
        encoder_weights = {
            key[len("module."):]: value
            for key, value in checkpoint.items()
            if key.startswith("module.encoder.") or key.startswith("module.codebook.")
        }
        self._vqvae.load_state_dict(encoder_weights, strict=False)
        self._vqvae.to(self.device).eval()

    def _load_sample_full_coords(self, *, sample: Any, bag_id: str) -> np.ndarray:
        """Load concatenated ``(N, 5)`` patch coordinates for the full sample."""
        coords_parts: list[np.ndarray] = []
        for artifact_path in list(getattr(sample, "artifact_paths", []) or []):
            with FileHandleH5(artifact_path, mode="r") as slide_artifact:
                coords_parts.append(
                    np.asarray(
                        tiles_io.read_coords(slide_artifact, bag_id=bag_id),
                        dtype=np.int32,
                    )
                )
        if not coords_parts:
            return np.empty((0, 5), dtype=np.int32)
        return np.concatenate(coords_parts, axis=0).astype(np.int32, copy=False)

    def _resolve_selected_indices(
        self,
        *,
        representation: RetrievalRepresentation,
        full_coords: np.ndarray,
    ) -> np.ndarray:
        """Resolve row indices of the selected patch subset within the full bag."""
        additional_data = representation.additional_data
        selected_indices = additional_data.get("selected_indices")
        features = np.asarray(representation.data)

        if selected_indices is not None:
            selected_indices = np.asarray(selected_indices, dtype=np.int64)
        elif features.shape[0] == full_coords.shape[0]:
            selected_indices = np.arange(full_coords.shape[0], dtype=np.int64)
        else:
            selected_indices = self._match_selected_coords_to_full_coords(
                selected_coords=np.asarray(
                    additional_data.get("selected_coords"),
                    dtype=np.int64,
                ),
                full_coords=full_coords[:, :2],
            )

        if selected_indices.ndim != 1:
            raise ValueError(
                "SISH selected indices must have shape (K,). "
                f"Got {selected_indices.shape}."
            )
        if selected_indices.shape[0] != features.shape[0]:
            raise ValueError(
                "SISH selected indices must align with representation rows. "
                f"Got selected_indices={selected_indices.shape[0]} and features={features.shape[0]}."
            )
        return selected_indices.astype(np.int64, copy=False)

    def _validate_selected_coords(
        self,
        *,
        representation: RetrievalRepresentation,
        selected_full_coords: np.ndarray,
    ) -> None:
        """Validate stored selected coordinates against the H5-backed full coords."""
        stored_coords = representation.additional_data.get("selected_coords")
        if stored_coords is None:
            return
        stored_coords = np.asarray(stored_coords, dtype=np.int64)
        if stored_coords.shape != selected_full_coords.shape:
            raise ValueError(
                "Stored selected_coords do not match resolved SISH coords shape. "
                f"Got {stored_coords.shape} and {selected_full_coords.shape}."
            )
        if not np.array_equal(stored_coords, selected_full_coords):
            raise ValueError(
                "selected_coords are not aligned with the selected patch rows in the H5 coords."
            )

    def _match_selected_coords_to_full_coords(
        self,
        *,
        selected_coords: np.ndarray,
        full_coords: np.ndarray,
    ) -> np.ndarray:
        """Resolve selected row indices by matching selected coords to full coords."""
        if selected_coords.ndim != 2 or selected_coords.shape[1] != 2:
            raise ValueError(
                "selected_coords are required with shape (K, 2) when selected_indices are absent."
            )

        coord_to_rows: dict[tuple[int, int], list[int]] = {}
        for row_index, coord in enumerate(np.asarray(full_coords, dtype=np.int64)):
            key = (int(coord[0]), int(coord[1]))
            coord_to_rows.setdefault(key, []).append(row_index)

        resolved_rows: list[int] = []
        for coord in np.asarray(selected_coords, dtype=np.int64):
            key = (int(coord[0]), int(coord[1]))
            rows = coord_to_rows.get(key)
            if not rows:
                raise ValueError(
                    "Could not match selected_coords to the full bag coordinate rows."
                )
            resolved_rows.append(int(rows.pop(0)))

        return np.asarray(resolved_rows, dtype=np.int64)

    def _build_patch_specs(
        self,
        *,
        sample: Any,
        bag_id: str,
        full_coords: np.ndarray,
        selected_indices: np.ndarray,
    ) -> list[_SelectedPatchSpec]:
        """Build one backend-read spec per selected patch."""
        slide_ids = [str(slide_id) for slide_id in list(getattr(sample, "slide_ids", []) or [])]
        artifact_paths = [Path(path) for path in list(getattr(sample, "artifact_paths", []) or [])]
        slide_paths = _resolve_sample_slide_paths(sample=sample, config=self.config)

        specs: list[_SelectedPatchSpec] = []
        start = 0
        intervals: list[tuple[int, int, str, Path]] = []
        for slide_id, artifact_path in zip(slide_ids, artifact_paths, strict=False):
            with FileHandleH5(artifact_path, mode="r") as slide_artifact:
                num_rows = int(tiles_io.coords_num_rows(slide_artifact, bag_id=bag_id))
            end = start + num_rows
            intervals.append((start, end, slide_id, artifact_path))
            start = end

        for output_position, selected_index in enumerate(selected_indices.tolist()):
            slide_id: str | None = None
            artifact_path: Path | None = None
            local_index: int | None = None
            for begin, end, current_slide_id, current_artifact_path in intervals:
                if begin <= selected_index < end:
                    slide_id = current_slide_id
                    artifact_path = current_artifact_path
                    local_index = int(selected_index - begin)
                    break
            if slide_id is None or artifact_path is None or local_index is None:
                raise ValueError(
                    f"Selected patch index {selected_index} could not be mapped to a slide."
                )

            coord_row = full_coords[int(selected_index)]
            slide_path = slide_paths.get(slide_id)
            if slide_path is None:
                raise FileNotFoundError(
                    f"Could not resolve source slide path for slide '{slide_id}'."
                )
            specs.append(
                _SelectedPatchSpec(
                    output_position=output_position,
                    slide_id=slide_id,
                    artifact_path=artifact_path,
                    slide_path=slide_path,
                    x=int(coord_row[0]),
                    y=int(coord_row[1]),
                    read_w=int(coord_row[2]),
                    read_h=int(coord_row[3]),
                    level=int(coord_row[4]),
                )
            )

        return specs

    def _encode_selected_patch_specs(
        self,
        *,
        patch_specs: list[_SelectedPatchSpec],
    ) -> np.ndarray:
        """Read selected patches, run the VQ-VAE, and return SISH indices."""
        self._load_models()
        if self._slide_processor is None:
            self._slide_processor = _build_slide_processor(config=self.config)

        if not patch_specs:
            return np.empty((0,), dtype=np.int64)

        output_latents: list[np.ndarray | None] = [None] * len(patch_specs)
        specs_by_slide: dict[tuple[str, Path, Path], list[_SelectedPatchSpec]] = {}
        for spec in patch_specs:
            key = (spec.slide_id, spec.artifact_path, spec.slide_path)
            specs_by_slide.setdefault(key, []).append(spec)

        batch_size = int(
            self._get_config_value(
                [
                    ("experiment", "sish", "vqvae_batch_size"),
                    ("experiment", "SISH_metrics", "vqvae_batch_size"),
                    ("sish", "vqvae_batch_size"),
                ],
                default=8,
            )
        )

        for (slide_id, artifact_path, slide_path), slide_specs in specs_by_slide.items():
            wsi = WSI(
                slide=slide_id,
                patient="",
                category="",
                path=slide_path,
                artifact_path=artifact_path,
            )
            self._slide_processor.load_wsi(wsi)
            try:
                tensors: list[torch.Tensor] = []
                output_positions: list[int] = []
                for spec in slide_specs:
                    patch = self._slide_processor.read_patch_region(
                        wsi,
                        x=spec.x,
                        y=spec.y,
                        width=spec.read_w,
                        height=spec.read_h,
                        level=spec.level,
                    )
                    patch_array = np.asarray(patch, dtype=np.uint8)
                    if patch_array.ndim != 3 or patch_array.shape[2] != 3:
                        raise ValueError(
                            "Patch RGB reads must have shape (H,W,3). "
                            f"Got {patch_array.shape} for slide '{slide_id}'."
                        )

                    tensor = torch.from_numpy(patch_array).permute(2, 0, 1).float() / 255.0
                    tensor = _scale_to_minus1_to_1(tensor)
                    tensors.append(tensor)
                    output_positions.append(spec.output_position)

                for start in range(0, len(tensors), batch_size):
                    batch_tensors = torch.stack(tensors[start : start + batch_size], dim=0)
                    batch_tensors = batch_tensors.to(self.device)
                    with torch.no_grad():
                        batch_latents = self._vqvae(batch_tensors).detach().cpu().numpy()
                    for offset, latent in enumerate(batch_latents):
                        output_latents[output_positions[start + offset]] = latent
            finally:
                self._slide_processor.close_wsi(wsi)

        if any(latent is None for latent in output_latents):
            raise RuntimeError("Failed to materialize VQ-VAE latents for all selected patches.")

        latent_array = np.stack([latent for latent in output_latents if latent is not None], axis=0)
        return _slide_to_index(
            latent_array,
            self._codebook_semantic,
            pool_layers=self._pool_layers,
        ).astype(np.int64, copy=False)

    def _pack_bits(self, features: Any) -> np.ndarray:
        """Pack foundation feature signs into one uint8 matrix for storage."""
        feature_array = np.asarray(features, dtype=np.float32)
        if feature_array.ndim != 2:
            raise ValueError(
                "SISH packed-bit export expects a 2D feature matrix. "
                f"Got {feature_array.shape}."
            )
        return np.packbits((feature_array > 0).astype(np.uint8, copy=False), axis=1)

    def _resolve_path(self, candidate_paths: list[tuple[str, ...]]) -> Path | None:
        """Resolve the first available filesystem path from config."""
        value = self._get_config_value(candidate_paths, default=None)
        if value is None:
            return None
        return Path(value)

    def _get_config_value(
        self,
        candidate_paths: list[tuple[str, ...]],
        *,
        default: Any,
    ) -> Any:
        """Resolve a nested config value using mapping or attribute access."""
        for path in candidate_paths:
            current = self.config
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
