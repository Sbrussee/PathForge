"""Unit tests for the lightweight registry implementation."""

from __future__ import annotations

import pytest

from pathforge.utils import registries as registries_module
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


def test_dynamic_registry_population_imports_builtin_trainers_once(monkeypatch) -> None:
    """The central bootstrap imports built-in trainers before policy lookup."""

    imported_modules: list[str] = []
    monkeypatch.setattr(registries_module, "_populated", False)
    monkeypatch.setattr(
        registries_module,
        "_import_builtin_trainer_modules",
        lambda: imported_modules.append("pathforge.training.lightning"),
    )
    monkeypatch.setattr(registries_module, "_import_native_model_modules", lambda: None)
    monkeypatch.setattr(
        registries_module, "populate_pathforge_feature_extractors", lambda: None
    )
    monkeypatch.setattr(registries_module, "is_torchmil_available", lambda: False)
    monkeypatch.setattr(registries_module, "is_mil_lab_available", lambda: False)
    monkeypatch.setattr(registries_module, "is_torchmetrics_available", lambda: False)
    monkeypatch.setattr(registries_module, "is_torchsurv_available", lambda: False)

    registries_module.populate_dynamic_registries()
    registries_module.populate_dynamic_registries()

    assert imported_modules == ["pathforge.training.lightning"]
