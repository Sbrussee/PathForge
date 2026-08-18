"""Canonical image-crop contract for SISH VQ-VAE descriptors."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional


SISH_CANONICAL_TILE_PX = 1024
SISH_CANONICAL_TILE_MPP = 0.5
SISH_CROP_ALIGNMENT = "source_tile_center"


def sish_descriptor_contract(
    *,
    checkpoint_path: str | None = None,
    codebook_path: str | None = None,
) -> dict[str, object]:
    """Return the cache contract for canonical SISH latent descriptors.

    Inputs:
        checkpoint_path: Optional identity of the VQ-VAE checkpoint.
        codebook_path: Optional identity of the semantic codebook.

    Output:
        Attribute mapping used to validate a row-aligned descriptor cache.

    Example:
        ``sish_descriptor_contract(checkpoint_path="model.pt")``.
    """
    contract: dict[str, object] = {
        "crop_alignment": SISH_CROP_ALIGNMENT,
        "crop_mpp": SISH_CANONICAL_TILE_MPP,
        "crop_px": SISH_CANONICAL_TILE_PX,
    }
    if checkpoint_path is not None:
        contract["vqvae_checkpoint"] = checkpoint_path
    if codebook_path is not None:
        contract["codebook_semantic"] = codebook_path
    return contract


def read_canonical_sish_crop(
    *,
    slide_processor: Any,
    wsi: Any,
    x_level0: int,
    y_level0: int,
    source_tile_px: int,
    source_tile_mpp: float,
) -> np.ndarray:
    """Read the fixed SISH field centred on one configured source tile.

    Inputs:
        x_level0/y_level0: Level-0 source-tile origin.
        source_tile_px/source_tile_mpp: Source-tile geometry.

    Output:
        RGB ``uint8`` array with shape ``(1024, 1024, 3)`` at 0.5 mpp.

    Example:
        ``read_canonical_sish_crop(..., source_tile_px=256, source_tile_mpp=0.5)``.
    """
    base_mpp = float(slide_processor.get_base_mpp(wsi))
    if base_mpp <= 0:
        raise ValueError(f"SISH requires a positive base MPP, got {base_mpp}.")
    if base_mpp > SISH_CANONICAL_TILE_MPP:
        raise ValueError(
            "SISH requires source imagery at or finer than 0.5 mpp; "
            f"got base_mpp={base_mpp}."
        )

    source_width_level0 = float(source_tile_px) * float(source_tile_mpp) / base_mpp
    canonical_width_level0 = (
        float(SISH_CANONICAL_TILE_PX) * SISH_CANONICAL_TILE_MPP / base_mpp
    )
    centre_x = float(x_level0) + (source_width_level0 / 2.0)
    centre_y = float(y_level0) + (source_width_level0 / 2.0)
    width = max(1, int(round(canonical_width_level0)))
    left = int(round(centre_x - (width / 2.0)))
    top = int(round(centre_y - (width / 2.0)))
    patch = np.asarray(
        slide_processor.read_patch_region(
            wsi, x=left, y=top, width=width, height=width, level=0
        ),
        dtype=np.uint8,
    )
    if patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError(f"SISH crop must have shape (H,W,3). Got {patch.shape}.")
    if patch.shape[:2] == (SISH_CANONICAL_TILE_PX, SISH_CANONICAL_TILE_PX):
        return patch

    tensor = torch.from_numpy(patch).permute(2, 0, 1)[None].float()
    resized = functional.interpolate(
        tensor,
        size=(SISH_CANONICAL_TILE_PX, SISH_CANONICAL_TILE_PX),
        mode="bilinear",
        align_corners=False,
    )[0]
    return resized.permute(1, 2, 0).round().clamp(0, 255).byte().numpy()
