"""Interface tests for public API documentation and typing coverage."""

from __future__ import annotations

import ast
from pathlib import Path


SRC_ROOT = Path("src/pathforge")


def _iter_public_top_level_nodes() -> list[tuple[Path, ast.AST]]:
    nodes: list[tuple[Path, ast.AST]] = []
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name.startswith("_"):
                    continue
                nodes.append((path, node))
    return nodes


def test_public_top_level_symbols_have_docstrings() -> None:
    """Public classes and functions should have module-level contract docstrings."""
    missing = [
        f"{path.relative_to(SRC_ROOT)}:{node.name}"
        for path, node in _iter_public_top_level_nodes()
        if not ast.get_docstring(node)
    ]
    assert missing == []


def test_public_top_level_functions_have_parameter_and_return_annotations() -> None:
    """Public functions should annotate all non-``self`` parameters and their return type."""
    missing: list[str] = []

    for path, node in _iter_public_top_level_nodes():
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        missing_parts: list[str] = []
        arguments = (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        )
        for arg in arguments:
            if arg.arg in {"self", "cls"}:
                continue
            if arg.annotation is None:
                missing_parts.append(arg.arg)

        if node.returns is None:
            missing_parts.append("return")

        if missing_parts:
            missing.append(
                f"{path.relative_to(SRC_ROOT)}:{node.name} -> {', '.join(missing_parts)}"
            )

    assert missing == []
