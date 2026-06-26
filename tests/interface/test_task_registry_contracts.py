"""Interface-level contract tests for the core task registry and TaskBase."""

from __future__ import annotations

import inspect

import pytest

from pathforge.core.tasks.base import TaskBase
from pathforge.core.tasks.registry import (
    _TASK_REGISTRY,
    _normalize_task_name,
    build_task,
    get_task,
    get_task_allowed_dataset_uses,
    import_task_modules,
    is_task_available,
    list_tasks,
    register_task,
)


# ---------------------------------------------------------------------------
# TaskBase abstract surface
# ---------------------------------------------------------------------------


def test_task_base_execute_is_abstract() -> None:
    """TaskBase.execute must remain abstract so every concrete task implements it."""
    assert "execute" in TaskBase.__abstractmethods__


def test_task_base_exposes_grid_keys_classmethod() -> None:
    assert callable(TaskBase.get_grid_keys)
    assert inspect.ismethod(TaskBase.get_grid_keys.__func__) or callable(TaskBase.get_grid_keys)


def test_task_base_exposes_allowed_dataset_uses_classmethod() -> None:
    assert callable(TaskBase.get_allowed_dataset_uses)


def test_task_base_default_grid_keys_is_empty_list() -> None:
    assert TaskBase.grid_keys == []


def test_task_base_default_allowed_dataset_uses_is_none() -> None:
    assert TaskBase.allowed_dataset_uses is None


def test_task_base_constructor_signature() -> None:
    sig = inspect.signature(TaskBase.__init__)
    params = list(sig.parameters)
    assert params == ["self", "experiment"]


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def test_normalize_task_name_strips_and_lowercases() -> None:
    assert _normalize_task_name("  Classification  ") == "classification"
    assert _normalize_task_name("SURVIVAL_DISCRETE") == "survival_discrete"


def test_normalize_task_name_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _normalize_task_name("")

    with pytest.raises(ValueError, match="non-empty"):
        _normalize_task_name("   ")


def test_list_tasks_returns_sorted_names() -> None:
    import_task_modules()
    tasks = list_tasks()
    assert tasks == sorted(tasks)
    assert len(tasks) > 0


def test_import_task_modules_registers_all_known_tasks() -> None:
    import_task_modules()
    registered = set(list_tasks())
    expected = {"classification", "regression", "survival", "survival_discrete", "slide_retrieval"}
    assert expected <= registered, f"Missing tasks: {expected - registered}"


def test_get_task_returns_task_base_subclass() -> None:
    import_task_modules()
    for name in list_tasks():
        cls = get_task(name)
        assert issubclass(cls, TaskBase), f"{name!r} → {cls} is not a TaskBase subclass"


def test_get_task_raises_on_unknown_name() -> None:
    with pytest.raises(ValueError, match="not registered"):
        get_task("nonexistent_task_xyz")


def test_is_task_available_true_for_known_tasks() -> None:
    import_task_modules()
    assert is_task_available("classification")
    assert is_task_available("slide_retrieval")


def test_is_task_available_false_for_unknown_task() -> None:
    assert not is_task_available("no_such_task_abc")


def test_get_task_allowed_dataset_uses_returns_frozenset_or_none() -> None:
    import_task_modules()
    for name in list_tasks():
        uses = get_task_allowed_dataset_uses(name)
        assert uses is None or isinstance(uses, frozenset), (
            f"{name!r}: expected frozenset or None, got {type(uses)}"
        )


def test_register_task_raises_on_duplicate_name() -> None:
    sentinel = "test_duplicate_task_sentinel"

    @register_task(sentinel)
    class _FirstTask(TaskBase):
        def execute(self, combo_cfg, datasets_by_use):
            return {}

    with pytest.raises(ValueError, match="already registered"):
        @register_task(sentinel)
        class _SecondTask(TaskBase):
            def execute(self, combo_cfg, datasets_by_use):
                return {}

    del _TASK_REGISTRY[sentinel]


def test_build_task_instantiates_task_with_experiment() -> None:
    from types import SimpleNamespace

    import_task_modules()

    fake_cfg = SimpleNamespace(
        experiment=SimpleNamespace(aggregation_level="slide"),
        slide_retrieval=None,
    )
    fake_experiment = SimpleNamespace(cfg=fake_cfg, project_root="/tmp")

    task = build_task("slide_retrieval", fake_experiment)

    assert isinstance(task, TaskBase)
    assert task.experiment is fake_experiment
    assert task.cfg is fake_cfg


# ---------------------------------------------------------------------------
# Concrete task invariants
# ---------------------------------------------------------------------------


def test_all_mil_tasks_define_non_empty_grid_keys() -> None:
    import_task_modules()
    mil_task_names = {"classification", "regression", "survival", "survival_discrete"}
    for name in mil_task_names:
        cls = get_task(name)
        assert cls.grid_keys, f"{name!r} task has empty grid_keys"


def test_mil_tasks_allowed_dataset_uses_are_training_roles() -> None:
    import_task_modules()
    expected_mil_uses = frozenset({"training", "validation", "testing", "all"})
    for name in ("classification", "regression", "survival", "survival_discrete"):
        uses = get_task_allowed_dataset_uses(name)
        assert uses == expected_mil_uses, (
            f"{name!r}: expected {expected_mil_uses}, got {uses}"
        )


def test_slide_retrieval_task_allowed_uses_are_retrieval_roles() -> None:
    import_task_modules()
    expected = frozenset({"reference", "query", "query_reference"})
    uses = get_task_allowed_dataset_uses("slide_retrieval")
    assert uses == expected


def test_slide_retrieval_task_grid_keys_include_retrieval_axes() -> None:
    import_task_modules()
    cls = get_task("slide_retrieval")
    grid_keys = set(cls.grid_keys)
    assert "retrieval_representation" in grid_keys
    assert "search_strategy" in grid_keys
    assert "feature_extraction" in grid_keys
