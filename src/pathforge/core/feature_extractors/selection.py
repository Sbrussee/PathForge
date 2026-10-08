"""Build-free selection of extractors for a slide-processing backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FeatureOutputLevel = Literal["patch", "slide"]


def determine_feature_extractor_output_level(name: str) -> FeatureOutputLevel:
    """Determine the requested embedding level using PathForge's naming policy.

    Args:
        name: Configured extractor name; a final ``-slide`` requests slide output.
    Returns:
        ``"patch"`` for ``(N, D)`` embeddings or ``"slide"`` for ``(1, D)``.
        Backend model capabilities do not override the requested output level.
    Example:
        >>> determine_feature_extractor_output_level("titan-slide")
        'slide'
    """
    # Keep output policy here so callers share one decision mechanism.
    return "slide" if name.endswith("-slide") else "patch"


def parse_feature_extractor_name(name: str) -> str:
    """Remove PathForge's output-selection suffix to obtain the backend model key.

    Args:
        name: Configured extractor name, optionally ending in ``-slide``.
    Returns:
        The model key with only the final ``-slide`` suffix removed.
    Example:
        >>> parse_feature_extractor_name("titan-slide")
        'titan'
    """
    return name.removesuffix("-slide")


FeatureExtractorSource = Literal["processor-native", "pathforge-native"]


@dataclass(frozen=True, slots=True)
class FeatureExtractorSelection:
    """A validated extractor choice and expected patch/slide output level.

    ``name`` retains the configured storage identity, including ``-slide``.
    ``output_level`` describes ``(N, D)`` patch or ``(1, D)`` slide embeddings.

    Example:
        >>> selection = FeatureExtractorSelection("titan-slide", "processor-native", "slide")
        >>> selection.output_level
        'slide'
    """

    name: str
    source: FeatureExtractorSource
    output_level: FeatureOutputLevel = "patch"

    @property
    def model_name(self) -> str:
        """Return the backend key without PathForge's output-selection suffix.

        Example:
            >>> FeatureExtractorSelection("titan-slide", "processor-native", "slide").model_name
            'titan'
        """
        return parse_feature_extractor_name(self.name)

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
    output_level = determine_feature_extractor_output_level(name)
    native_names = processor.native_feature_extractor_names()
    if name in native_names:
        return FeatureExtractorSelection(
            name=name,
            source="processor-native",
            output_level=output_level,
        )
    if (
        output_level == "patch"
        and processor.supports_pathforge_feature_extractors()
        and is_native_feature_extractor_available(name)
    ):
        return FeatureExtractorSelection(name=name, source="pathforge-native")

    available_names = set(native_names)
    if processor.supports_pathforge_feature_extractors():
        available_names |= registered_feature_extractor_names()
    raise ValueError(
        f"Feature extractor '{name}' is not available for slide processing "
        f"backend '{backend_name}'. Available feature extractors: {sorted(available_names)}"
    )
