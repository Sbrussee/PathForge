from __future__ import annotations

from pathforge.core.experiments.combinations import ComboConfig


def build_tiling_id(combo_cfg: ComboConfig) -> str:
    """Build the canonical tiling identifier from a combination config."""
    tile_px = int(combo_cfg.tile_px)
    tile_mpp = float(combo_cfg.tile_mpp)
    return f"{tile_px}px_{tile_mpp:g}mpp"


def build_bag_id(combo_cfg: ComboConfig) -> str:
    """Build the canonical bag identifier from a combination config."""
    tiling_id = build_tiling_id(combo_cfg)
    feature_name = build_feature_name(combo_cfg)
    return f"{tiling_id}__{feature_name}"


def build_feature_name(combo_cfg: ComboConfig) -> str:
    """Build the canonical stored feature name from a combination config."""
    feature_extraction = str(combo_cfg.feature_extraction).strip()
    if not feature_extraction:
        raise ValueError("feature_extraction must be a non-empty string.")

    color_norm = combo_cfg.get("color_norm")
    normalized_color_norm = None if color_norm is None else str(color_norm).strip()
    if not normalized_color_norm:
        return feature_extraction

    return f"{feature_extraction}_{normalized_color_norm}"
