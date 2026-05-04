"""Architecture-boundary tests for the PathBench package layers."""

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
    """The core layer must not depend on outer runtime layers."""
    core_root = Path("src/pathbench/core")
    forbidden_prefixes = (
        "pathbench.cli",
        "pathbench.policy",
        "pathbench.training",
        "pathbench.inference",
        "pathbench.optimization",
        "pathbench.benchmarking",
    )
    offenders: list[tuple[str, str]] = []

    for path in core_root.rglob("*.py"):
        for module_name in _imported_modules(path):
            if module_name.startswith(forbidden_prefixes):
                offenders.append((str(path), module_name))

    assert offenders == []


def test_policy_layer_does_not_import_cli_layer() -> None:
    """Policies should stay independent from CLI entrypoints."""
    policy_root = Path("src/pathbench/policy")
    offenders: list[tuple[str, str]] = []

    for path in policy_root.rglob("*.py"):
        for module_name in _imported_modules(path):
            if module_name.startswith("pathbench.cli"):
                offenders.append((str(path), module_name))

    assert offenders == []


def test_training_layer_does_not_import_cli_layer() -> None:
    """Training code should not depend on CLI entrypoints."""
    training_root = Path("src/pathbench/training")
    offenders: list[tuple[str, str]] = []

    for path in training_root.rglob("*.py"):
        for module_name in _imported_modules(path):
            if module_name.startswith("pathbench.cli"):
                offenders.append((str(path), module_name))

    assert offenders == []
