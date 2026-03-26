from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class H5Layout:
    # ---- shared (per artifact) ----
    tissue_dataset: str = "annotations/tissue"  # scalar UTF-8 JSON string

    # ---- bags (per tiling) ----
    bags_group: str = "bags"
    coords_name: str = "coords"  # (N,5) int32
    tiling_spec_name: str = "tiling_spec"  # scalar UTF-8 JSON string
    tiles_overview_name: str = "tiles_overview"  # JPEG bytes stored as uint8 vector
    features_group_name: str = "features"  # features/{extractor} -> (N,D) float32
    descriptors_group_name: str = "descriptors"  # descriptors/{name} -> (N,D) float32

    # ---- retrieval representations (per tiling) ----
    retrieval_representations_group_name: str = "retrieval_representations"
    retrieval_representation_sample_id_name: str = "sample_id"  # scalar UTF-8 string
    retrieval_representation_type_name: str = "representation_type"  # scalar UTF-8 string
    retrieval_representation_metadata_name: str = "metadata"  # scalar UTF-8 JSON string
    retrieval_representation_params_name: str = "params"  # scalar UTF-8 JSON string
    retrieval_representation_slide_ids_name: str = "slide_ids"  # (N,) UTF-8 strings
    retrieval_representation_data_group_name: str = "data"
    retrieval_representation_main_name: str = "main"  # main representation array
    retrieval_representation_additional_data_group_name: str = "additional_data"

    # ------------------------------------------------------------------
    # Bags
    # ------------------------------------------------------------------

    def bag_group(self, bag_id: str) -> str:
        _validate_name(bag_id, "bag_id")
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
        _validate_name(extractor_name, "extractor_name")
        return f"{self.features_group(bag_id)}/{extractor_name}"

    def descriptors_group(self, bag_id: str) -> str:
        return f"{self.bag_group(bag_id)}/{self.descriptors_group_name}"

    def descriptor_dataset(self, bag_id: str, descriptor_name: str) -> str:
        _validate_name(descriptor_name, "descriptor_name")
        return f"{self.descriptors_group(bag_id)}/{descriptor_name}"

    # ------------------------------------------------------------------
    # Retrieval representations
    # ------------------------------------------------------------------

    def retrieval_representations_group(self, bag_id: str) -> str:
        return f"{self.bag_group(bag_id)}/{self.retrieval_representations_group_name}"

    def retrieval_representation_group(self, bag_id: str, representation_id: str) -> str:
        _validate_name(representation_id, "representation_id")
        return f"{self.retrieval_representations_group(bag_id)}/{representation_id}"

    def retrieval_representation_entry_group(
        self,
        bag_id: str,
        representation_id: str,
        entry_id: str,
    ) -> str:
        _validate_name(entry_id, "entry_id")
        return f"{self.retrieval_representation_group(bag_id, representation_id)}/{entry_id}"

    def retrieval_representation_metadata_dataset(
        self,
        bag_id: str,
        representation_id: str,
        entry_id: str,
    ) -> str:
        return (
            f"{self.retrieval_representation_entry_group(bag_id, representation_id, entry_id)}"
            f"/{self.retrieval_representation_metadata_name}"
        )

    def retrieval_representation_sample_id_dataset(
        self,
        bag_id: str,
        representation_id: str,
        entry_id: str,
    ) -> str:
        return (
            f"{self.retrieval_representation_entry_group(bag_id, representation_id, entry_id)}"
            f"/{self.retrieval_representation_sample_id_name}"
        )

    def retrieval_representation_type_dataset(
        self,
        bag_id: str,
        representation_id: str,
        entry_id: str,
    ) -> str:
        return (
            f"{self.retrieval_representation_entry_group(bag_id, representation_id, entry_id)}"
            f"/{self.retrieval_representation_type_name}"
        )

    def retrieval_representation_params_dataset(
        self,
        bag_id: str,
        representation_id: str,
        entry_id: str,
    ) -> str:
        return (
            f"{self.retrieval_representation_entry_group(bag_id, representation_id, entry_id)}"
            f"/{self.retrieval_representation_params_name}"
        )

    def retrieval_representation_slide_ids_dataset(
        self,
        bag_id: str,
        representation_id: str,
        entry_id: str,
    ) -> str:
        return (
            f"{self.retrieval_representation_entry_group(bag_id, representation_id, entry_id)}"
            f"/{self.retrieval_representation_slide_ids_name}"
        )

    def retrieval_representation_data_group(
        self,
        bag_id: str,
        representation_id: str,
        entry_id: str,
    ) -> str:
        return (
            f"{self.retrieval_representation_entry_group(bag_id, representation_id, entry_id)}"
            f"/{self.retrieval_representation_data_group_name}"
        )

    def retrieval_representation_main_dataset(
        self,
        bag_id: str,
        representation_id: str,
        entry_id: str,
    ) -> str:
        return (
            f"{self.retrieval_representation_data_group(bag_id, representation_id, entry_id)}"
            f"/{self.retrieval_representation_main_name}"
        )

    def retrieval_representation_additional_data_group(
        self,
        bag_id: str,
        representation_id: str,
        entry_id: str,
    ) -> str:
        return (
            f"{self.retrieval_representation_entry_group(bag_id, representation_id, entry_id)}"
            f"/{self.retrieval_representation_additional_data_group_name}"
        )

    def retrieval_representation_additional_data_dataset(
        self,
        bag_id: str,
        representation_id: str,
        entry_id: str,
        name: str,
    ) -> str:
        _validate_name(name, "additional_data_name")
        return (
            f"{self.retrieval_representation_additional_data_group(bag_id, representation_id, entry_id)}"
            f"/{name}"
        )


def _validate_name(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    if "/" in value:
        raise ValueError(f"{field_name} may not contain '/': {value!r}")


DEFAULT_LAYOUT = H5Layout()
