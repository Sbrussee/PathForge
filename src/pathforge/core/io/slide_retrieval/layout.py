from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalH5Layout:
    """Build canonical dataset paths inside a retrieval-artifact H5 file.

    Per-tiling descriptors and representations live below ``bags/{tile_id}``.
    Slide representations are stored directly below their representation ID;
    multi-slide aggregations add an entry ID for the concrete member set.
    """

    retrieval_representations_group_name: str = "retrieval_representations"
    descriptors_group_name: str = "descriptors"  # descriptors/{name} -> (N,D) float32
    retrieval_representation_type_name: str = "representation_type"  # scalar UTF-8 string
    retrieval_representation_metadata_name: str = "metadata"  # scalar UTF-8 JSON string
    retrieval_representation_params_name: str = "params"  # scalar UTF-8 JSON string
    retrieval_representation_embedding_name: str = "embedding"  # main representation array
    retrieval_representation_additional_data_group_name: str = "additional_data"

    def tile_group(self, tile_id: str) -> str:
        """Return the root group for one tiling configuration: `bags/{tile_id}`."""
        _validate_name(tile_id, "tile_id")
        return f"bags/{tile_id}"

    def retrieval_representations_group(self, tile_id: str) -> str:
        """Return the retrieval-representation container under one tile group."""
        return f"{self.tile_group(tile_id)}/{self.retrieval_representations_group_name}"

    def descriptors_group(self, tile_id: str) -> str:
        """Return the retrieval-side descriptor cache group under one tile group."""
        return f"{self.tile_group(tile_id)}/{self.descriptors_group_name}"

    def descriptor(self, tile_id: str, descriptor_name: str) -> str:
        """Return one descriptor dataset: `bags/{tile_id}/descriptors/{descriptor_name}`."""
        _validate_name(descriptor_name, "descriptor_name")
        return f"{self.descriptors_group(tile_id)}/{descriptor_name}"

    def retrieval_representation_group(self, tile_id: str, representation_id: str) -> str:
        """Return the group for one retrieval configuration under one tile group."""
        _validate_name(representation_id, "representation_id")
        return f"{self.retrieval_representations_group(tile_id)}/{representation_id}"

    def retrieval_representation_entry_group(
        self,
        tile_id: str,
        representation_id: str,
        entry_id: str | None,
    ) -> str:
        """Return the group for one stored retrieval entry under one representation.

        When `entry_id` is `None`, the representation root itself is used.
        """
        if entry_id is None:
            return self.retrieval_representation_group(tile_id, representation_id)
        _validate_name(entry_id, "entry_id")
        return f"{self.retrieval_representation_group(tile_id, representation_id)}/{entry_id}"

    def retrieval_representation_metadata(
        self,
        tile_id: str,
        representation_id: str,
        entry_id: str | None,
    ) -> str:
        """Return the JSON metadata dataset for one retrieval entry."""
        return (
            f"{self.retrieval_representation_entry_group(tile_id, representation_id, entry_id)}"
            f"/{self.retrieval_representation_metadata_name}"
        )

    def retrieval_representation_type(
        self,
        tile_id: str,
        representation_id: str,
        entry_id: str | None,
    ) -> str:
        """Return the representation-type dataset for one retrieval entry."""
        return (
            f"{self.retrieval_representation_entry_group(tile_id, representation_id, entry_id)}"
            f"/{self.retrieval_representation_type_name}"
        )

    def retrieval_representation_params(
        self,
        tile_id: str,
        representation_id: str,
        entry_id: str | None,
    ) -> str:
        """Return the readable params dataset for one retrieval entry."""
        return (
            f"{self.retrieval_representation_entry_group(tile_id, representation_id, entry_id)}"
            f"/{self.retrieval_representation_params_name}"
        )

    def retrieval_representation_embedding(
        self,
        tile_id: str,
        representation_id: str,
        entry_id: str | None,
    ) -> str:
        """Return the main embedding dataset for one retrieval entry."""
        return (
            f"{self.retrieval_representation_entry_group(tile_id, representation_id, entry_id)}"
            f"/{self.retrieval_representation_embedding_name}"
        )

    def retrieval_representation_additional_data_group(
        self,
        tile_id: str,
        representation_id: str,
        entry_id: str | None,
    ) -> str:
        """Return the group that stores extra strategy-specific arrays."""
        return (
            f"{self.retrieval_representation_entry_group(tile_id, representation_id, entry_id)}"
            f"/{self.retrieval_representation_additional_data_group_name}"
        )

    def retrieval_representation_additional_data(
        self,
        tile_id: str,
        representation_id: str,
        entry_id: str | None,
        name: str,
    ) -> str:
        """Return one dataset inside `additional_data/` for a retrieval entry."""
        _validate_name(name, "additional_data_name")
        return (
            f"{self.retrieval_representation_additional_data_group(tile_id, representation_id, entry_id)}"
            f"/{name}"
        )


def _validate_name(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    if "/" in value:
        raise ValueError(f"{field_name} may not contain '/': {value!r}")


DEFAULT_LAYOUT = RetrievalH5Layout()
