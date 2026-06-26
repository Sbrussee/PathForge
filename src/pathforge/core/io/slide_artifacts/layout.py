from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class H5Layout:
    """
    Path builder for slide-artifact H5 files stored in `artifacts_dir`.

    One slide artifact file lives at:
        artifacts_dir/{slide_id}.h5

    Inside that file, data is organized as:

        annotations/
          tissue

        thumbnail/
          image
          spec

        bags/
          {bag_id}/
            coords
            tiling_spec
            tiles_overview
            features/
              {extractor_name}

    Meanings:
    - `bag_id`:
        Canonical tiling identifier, e.g. `256px_0.5mpp`.
    - `tissue`:
        Slide-level tissue annotation payload.
    - `coords`:
        `(N, 5)` patch coordinate matrix for one tiling configuration.
    - `tiling_spec`:
        JSON payload describing the tiling setup used to create `coords`.
    - `tiles_overview`:
        Optional rendered overview image for one tiling configuration.
    - `extractor_name`:
        Canonical stored feature name for one feature extractor configuration.
    """

    # ---- shared (per artifact) ----
    tissue_dataset: str = "annotations/tissue"  # scalar UTF-8 JSON string
    thumbnail_group: str = "thumbnail"
    thumbnail_image_name: str = "image"  # JPEG bytes stored as uint8 vector
    thumbnail_spec_name: str = "spec"  # scalar UTF-8 JSON string

    # ---- bags (per tiling) ----
    bags_group: str = "bags"
    coords_name: str = "coords"  # (N,5) int32
    tiling_spec_name: str = "tiling_spec"  # scalar UTF-8 JSON string
    tiles_overview_name: str = "tiles_overview"  # JPEG bytes stored as uint8 vector
    features_group_name: str = "features"  # features/{extractor} -> (N,D) float32

    # ------------------------------------------------------------------
    # Bags
    # ------------------------------------------------------------------

    def bag_group(self, bag_id: str) -> str:
        """Return the root group for one tiling configuration: `bags/{bag_id}`."""
        _validate_name(bag_id, "bag_id")
        return f"{self.bags_group}/{bag_id}"

    def coords_dataset(self, bag_id: str) -> str:
        """Return the coords dataset for one bag."""
        return f"{self.bag_group(bag_id)}/{self.coords_name}"

    # ------------------------------------------------------------------
    # Thumbnail
    # ------------------------------------------------------------------

    def thumbnail_image_dataset(self) -> str:
        """Return the slide-level thumbnail image dataset."""
        return f"{self.thumbnail_group}/{self.thumbnail_image_name}"

    def thumbnail_spec_dataset(self) -> str:
        """Return the slide-level thumbnail spec dataset."""
        return f"{self.thumbnail_group}/{self.thumbnail_spec_name}"

    def tiling_spec_dataset(self, bag_id: str) -> str:
        """Return the tiling-spec dataset for one bag."""
        return f"{self.bag_group(bag_id)}/{self.tiling_spec_name}"

    def tiles_overview_dataset(self, bag_id: str) -> str:
        """Return the tiles-overview dataset for one bag."""
        return f"{self.bag_group(bag_id)}/{self.tiles_overview_name}"

    def features_group(self, bag_id: str) -> str:
        """Return the features group for one bag."""
        return f"{self.bag_group(bag_id)}/{self.features_group_name}"

    def features_dataset(self, bag_id: str, extractor_name: str) -> str:
        """Return one feature dataset under `features/{extractor_name}`."""
        _validate_name(extractor_name, "extractor_name")
        return f"{self.features_group(bag_id)}/{extractor_name}"


def _validate_name(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    if "/" in value:
        raise ValueError(f"{field_name} may not contain '/': {value!r}")


DEFAULT_LAYOUT = H5Layout()
