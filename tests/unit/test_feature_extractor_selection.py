"""PathForge owns output selection independently of backend model defaults."""

import pytest

from pathforge.core.feature_extractors import selection as selection_module
from pathforge.core.feature_extractors.selection import (
    FeatureExtractorSelection,
    determine_feature_extractor_output_level,
    parse_feature_extractor_name,
)


@pytest.mark.parametrize(
    ("configured", "model_name", "level"),
    [
        ("titan", "titan", "patch"),
        ("titan-slide", "titan", "slide"),
        ("conch_v1.5", "conch_v1.5", "patch"),
        ("gigapath-slide-encoder", "gigapath-slide-encoder", "patch"),
        ("gigapath-slide-encoder-slide", "gigapath-slide-encoder", "slide"),
        ("slide", "slide", "patch"),
        ("titan-slide-slide", "titan-slide", "slide"),
        ("titan-SLIDE", "titan-SLIDE", "patch"),
        ("", "", "patch"),
    ],
)
def test_pathforge_parses_only_the_final_slide_suffix(configured, model_name, level):
    """Backend family names and internal occurrences do not determine output."""
    assert determine_feature_extractor_output_level(configured) == level
    assert parse_feature_extractor_name(configured) == model_name
    selection = FeatureExtractorSelection(configured, "processor-native", level)
    assert selection.name == configured
    assert selection.model_name == model_name
    assert selection.output_level == level


def test_selection_carries_explicit_slide_request_without_a_configured_suffix():
    """Backends can execute an explicit request without interpreting YAML names."""
    selection = FeatureExtractorSelection("titan", "processor-native", "slide")
    assert selection.model_name == "titan"
    assert selection.output_level == "slide"


def test_model_name_parsing_is_independent_of_output_policy(monkeypatch):
    """Changing output determination must not change the backend model key."""
    monkeypatch.setattr(
        selection_module,
        "determine_feature_extractor_output_level",
        lambda name: "patch",
    )
    assert parse_feature_extractor_name("titan-slide") == "titan"
    assert FeatureExtractorSelection("titan-slide", "processor-native").model_name == "titan"
