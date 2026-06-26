from __future__ import annotations

import pytest

from pathbench.slide_retrieval.representation_strategies.registry import (
    get_representation_strategy_output_kind,
    get_representation_strategy_supported_feature_levels,
)
from pathbench.slide_retrieval.search_strategies.registry import (
    get_search_strategy_supported_representation_kinds,
)


def test_get_representation_strategy_output_kind_returns_class_metadata() -> None:
    output_kind = get_representation_strategy_output_kind("splice-features")

    assert output_kind == "patch_vector"


def test_get_representation_strategy_output_kind_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="is not registered"):
        get_representation_strategy_output_kind("missing_representation")


def test_get_search_strategy_supported_kinds_returns_class_metadata() -> None:
    supported_kinds = get_search_strategy_supported_representation_kinds("sish")

    assert "patch_vector" in supported_kinds


def test_get_search_strategy_supported_kinds_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="is not registered"):
        get_search_strategy_supported_representation_kinds("missing_search")


def test_get_representation_strategy_supported_feature_levels_returns_metadata() -> None:
    feature_levels = get_representation_strategy_supported_feature_levels(
        "splice-features"
    )

    assert "patch" in feature_levels


def test_get_representation_strategy_supported_feature_levels_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="is not registered"):
        get_representation_strategy_supported_feature_levels("missing_representation")
