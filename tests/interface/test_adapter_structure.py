"""Architecture tests for concrete adapters, strategies, and trainers."""

from __future__ import annotations

import ast
from pathlib import Path


SRC_ROOT = Path("src/pathforge")


def _base_names(node: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            parts: list[str] = []
            current: ast.expr = base
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            names.add(".".join(reversed(parts)))
    return names


def _public_classes(path: Path) -> list[ast.ClassDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    ]


def test_concrete_trainers_subclass_trainer_base() -> None:
    """Concrete trainer implementations should be wired through ``TrainerBase``."""
    training_root = SRC_ROOT / "training"
    offenders: list[str] = []

    for path in training_root.rglob("*.py"):
        if path.name == "base.py":
            continue
        for node in _public_classes(path):
            if not node.name.endswith("Trainer"):
                continue
            if "TrainerBase" not in _base_names(node):
                offenders.append(f"{path.relative_to(SRC_ROOT)}:{node.name}")

    assert offenders == []


def test_search_and_representation_strategies_subclass_strategy_bases() -> None:
    """Retrieval strategy implementations should extend the declared strategy bases."""
    offenders: list[str] = []

    search_root = SRC_ROOT / "slide_retrieval" / "search_strategies" / "strategies"
    for path in search_root.rglob("*.py"):
        for node in _public_classes(path):
            if node.name.endswith("Search") and "BaseSearchStrategy" not in _base_names(
                node
            ):
                offenders.append(f"{path.relative_to(SRC_ROOT)}:{node.name}")

    representation_root = (
        SRC_ROOT / "slide_retrieval" / "representation_strategies" / "strategies"
    )
    for path in representation_root.rglob("*.py"):
        for node in _public_classes(path):
            if node.name.startswith(("SPLICE", "Yottixel")) or node.name.endswith(
                ("Features", "RGB")
            ):
                if "BaseRetrievalRepresentationStrategy" not in _base_names(node) and (
                    "_BaseSPLICEStrategy" not in _base_names(node)
                    and "_BaseYottixelRepresentationStrategy" not in _base_names(node)
                ):
                    offenders.append(f"{path.relative_to(SRC_ROOT)}:{node.name}")

    assert offenders == []


def test_evaluation_and_visualization_adapters_subclass_adapter_bases() -> None:
    """Task-specific adapter implementations should extend the adapter base classes."""
    offenders: list[str] = []

    evaluation_root = SRC_ROOT / "core" / "evaluation" / "tasks"
    for path in evaluation_root.rglob("*.py"):
        for node in _public_classes(path):
            if node.name.endswith("Adapter") and "TaskEvaluationAdapterBase" not in _base_names(
                node
            ):
                offenders.append(f"{path.relative_to(SRC_ROOT)}:{node.name}")

    visualization_root = SRC_ROOT / "core" / "visualization" / "tasks"
    for path in visualization_root.rglob("*.py"):
        for node in _public_classes(path):
            if node.name.endswith("Adapter") and "TaskVisualizationAdapterBase" not in _base_names(
                node
            ):
                offenders.append(f"{path.relative_to(SRC_ROOT)}:{node.name}")

    assert offenders == []
