"""Build-free selection of extractors for a slide-processing backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FeatureExtractorSource = Literal["processor-native", "pathforge-native"]


@dataclass(frozen=True, slots=True)
class FeatureExtractorSelection:
    """A validated extractor choice for one slide-processing backend."""

    name: str
    source: FeatureExtractorSource

    @property
    def requires_pathforge_adapter(self) -> bool:
        """Return whether runtime must adapt a PathForge extractor."""
        return self.source == "pathforge-native"


def resolve_feature_extractor_selection(
    backend_name: str,
    name: str,
) -> FeatureExtractorSelection:
    """Validate and select one name without constructing a model."""
    from pathforge.core.feature_extractors.factory import (
        is_native_feature_extractor_available,
        registered_feature_extractor_names,
    )
    from pathforge.core.slide_processing.factory import build_slide_processor
    from pathforge.utils.registries import populate_pathforge_feature_extractors

    populate_pathforge_feature_extractors()
    processor = build_slide_processor(backend_name)
    native_names = processor.native_feature_extractor_names()
    if name in native_names:
        return FeatureExtractorSelection(name=name, source="processor-native")
    if processor.supports_pathforge_feature_extractors() and is_native_feature_extractor_available(name):
        return FeatureExtractorSelection(name=name, source="pathforge-native")

    available_names = set(native_names)
    if processor.supports_pathforge_feature_extractors():
        available_names |= registered_feature_extractor_names()
    raise ValueError(
        f"Feature extractor '{name}' is not available for slide processing "
        f"backend '{backend_name}'. Available feature extractors: {sorted(available_names)}"
    )
