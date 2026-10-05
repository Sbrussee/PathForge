from __future__ import annotations

import pytest

from pathforge.slide_retrieval.representation_strategies.registry import (
    get_representation_strategy_output_kind,
    get_representation_strategy_supported_feature_levels,
    import_representation_strategy_modules,
    list_representation_strategies,
)
from pathforge.slide_retrieval.search_strategies.registry import (
    get_search_strategy_supported_representation_kinds,
    import_search_strategy_modules,
    list_search_strategies,
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


def test_patch_and_slide_families_remain_isolated_in_registries() -> None:
    import_representation_strategy_modules()
    import_search_strategy_modules()

    representation_names = list_representation_strategies()
    search_names = list_search_strategies()
    assert "slide_features" in representation_names
    assert "slide-barcode-faiss" in search_names

    for name in representation_names:
        levels = get_representation_strategy_supported_feature_levels(name)
        kind = get_representation_strategy_output_kind(name)
        if name == "slide_features":
            assert levels == {"slide"}
            assert kind == "slide_vector"
            continue
        assert "slide" not in levels
        assert kind != "slide_vector"

    for name in search_names:
        kinds = get_search_strategy_supported_representation_kinds(name)
        if name == "slide-barcode-faiss":
            assert kinds == {"slide_vector"}
            continue
        assert "slide_vector" not in kinds
