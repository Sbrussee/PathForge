# src/pathbench/core/io/h5/layout.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class H5Layout:
    """Canonical HDF5 path layout for per-slide PathBench artifacts."""

    # ---- shared (per slide) ----
    tissue_dataset: str = "annotations/tissue"  # scalar UTF-8 JSON string

    # ---- bags (per tiling) ----
    bags_group: str = "bags"
    coords_name: str = "coords"  # (N,5) int32
    tiling_spec_name: str = "tiling_spec"  # scalar UTF-8 JSON string
    tiles_overview_name: str = "tiles_overview"  # JPEG bytes stored as uint8 vector
    features_group_name: str = "features"  # features/{extractor} -> (N,D) float32
    predictions_group_name: str = "predictions"
    heatmaps_group_name: str = "heatmaps"
    heatmap_coords_name: str = "coords"  # (N,2) int32/float32 rendered coords
    heatmap_scores_name: str = "scores"  # (N,) float32 normalized scores
    heatmap_metadata_name: str = "metadata"  # scalar UTF-8 JSON string

    def bag_group(self, bag_id: str) -> str:
        _validate_bag_id(bag_id)
        return f"{self.bags_group}/{bag_id}"

    def coords_dataset(self, bag_id: str) -> str:
        return f"{self.bag_group(bag_id)}/{self.coords_name}"

    def tiling_spec_dataset(self, bag_id: str) -> str:
        return f"{self.bag_group(bag_id)}/{self.tiling_spec_name}"

    def tiles_overview_dataset(self, bag_id: str) -> str:
        return f"{self.bag_group(bag_id)}/{self.tiles_overview_name}"

    def features_group(self, bag_id: str) -> str:
        return f"{self.bag_group(bag_id)}/{self.features_group_name}"

    def features_dataset(self, bag_id: str, extractor_name: str) -> str:
        if not extractor_name or "/" in extractor_name:
            raise ValueError(f"Invalid extractor_name: {extractor_name!r}")
        return f"{self.features_group(bag_id)}/{extractor_name}"

    def prediction_heatmaps_group(self, bag_id: str) -> str:
        return f"{self.bag_group(bag_id)}/{self.predictions_group_name}/{self.heatmaps_group_name}"

    def prediction_heatmap_group(self, bag_id: str, heatmap_name: str) -> str:
        _validate_heatmap_name(heatmap_name)
        return f"{self.prediction_heatmaps_group(bag_id)}/{heatmap_name}"

    def prediction_heatmap_coords_dataset(self, bag_id: str, heatmap_name: str) -> str:
        return f"{self.prediction_heatmap_group(bag_id, heatmap_name)}/{self.heatmap_coords_name}"

    def prediction_heatmap_scores_dataset(self, bag_id: str, heatmap_name: str) -> str:
        return f"{self.prediction_heatmap_group(bag_id, heatmap_name)}/{self.heatmap_scores_name}"

    def prediction_heatmap_metadata_dataset(self, bag_id: str, heatmap_name: str) -> str:
        return f"{self.prediction_heatmap_group(bag_id, heatmap_name)}/{self.heatmap_metadata_name}"


def _validate_bag_id(bag_id: str) -> None:
    if not bag_id:
        raise ValueError("bag_id must be a non-empty string.")
    if "/" in bag_id:
        raise ValueError(f"bag_id may not contain '/': {bag_id!r}")


def _validate_heatmap_name(heatmap_name: str) -> None:
    if not heatmap_name:
        raise ValueError("heatmap_name must be a non-empty string.")
    if "/" in heatmap_name:
        raise ValueError(f"heatmap_name may not contain '/': {heatmap_name!r}")


DEFAULT_LAYOUT = H5Layout()
