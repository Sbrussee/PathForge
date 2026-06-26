"""Architecture-boundary tests for the PathForge package layers."""

from __future__ import annotations

import ast
from pathlib import Path


def _imported_modules(path: Path) -> set[str]:
    """Return imported module names for one parseable Python source file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_core_layer_does_not_import_outer_application_layers() -> None:
    """The core layer must not depend on outer runtime layers.

    core/tasks/ is excluded from this check because task implementations are
    use-case orchestrators that intentionally depend on policy utilities
    (apply_search_params, build_mil_model_for_config). They live in core/ for
    co-location with the registry and base class, but are not pure domain
    abstractions.
    """
    core_root = Path("src/pathforge/core")
    tasks_root = Path("src/pathforge/core/tasks")
    forbidden_prefixes = (
        "pathforge.cli",
        "pathforge.policy",
        "pathforge.training",
        "pathforge.inference",
        "pathforge.optimization",
    )
    offenders: list[tuple[str, str]] = []

    for path in core_root.rglob("*.py"):
        if path.is_relative_to(tasks_root):
            continue
        for module_name in _imported_modules(path):
            if module_name.startswith(forbidden_prefixes):
                offenders.append((str(path), module_name))

    assert offenders == []


def test_policy_layer_does_not_import_cli_layer() -> None:
    """Policies should stay independent from CLI entrypoints."""
    policy_root = Path("src/pathforge/policy")
    offenders: list[tuple[str, str]] = []

    for path in policy_root.rglob("*.py"):
        for module_name in _imported_modules(path):
            if module_name.startswith("pathforge.cli"):
                offenders.append((str(path), module_name))

    assert offenders == []


def test_training_layer_does_not_import_cli_layer() -> None:
    """Training code should not depend on CLI entrypoints."""
    training_root = Path("src/pathforge/training")
    offenders: list[tuple[str, str]] = []

    for path in training_root.rglob("*.py"):
        for module_name in _imported_modules(path):
            if module_name.startswith("pathforge.cli"):
                offenders.append((str(path), module_name))

    assert offenders == []
