"""Unit tests for the lightweight registry implementation."""

from __future__ import annotations

import pytest

from pathforge.utils.registry import Registry


def test_registry_register_get_and_list_plugins() -> None:
    registry = Registry()

    @registry.register("plugin_a")
    def plugin_a() -> int:
        return 1

    assert registry.is_available("plugin_a") is True
    assert registry.get("plugin_a")() == 1
    assert registry.list_plugins() == ["plugin_a"]


def test_registry_rejects_duplicate_registration() -> None:
    registry = Registry()

    @registry.register("plugin_a")
    def plugin_a() -> int:
        return 1

    with pytest.raises(KeyError, match="Duplicate registration"):

        @registry.register("plugin_a")
        def plugin_b() -> int:
            return 2


def test_registry_get_raises_for_missing_plugin() -> None:
    registry = Registry()

    with pytest.raises(KeyError, match="not found in registry"):
        registry.get("missing_plugin")
