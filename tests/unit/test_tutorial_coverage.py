"""Executable guardrails for every focused documentation tutorial."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_DIR = REPO_ROOT / "docs" / "tutorials"
WORKFLOW_COVERAGE = {
    "end_to_end.rst": [
        "tests/smoke/test_feature_extraction_smoke.py",
        "tests/unit/test_policy_benchmark.py",
        "tests/unit/test_model_package.py",
        "tests/unit/test_infer_cli.py",
    ],
    "feature_extraction.rst": ["tests/smoke/test_feature_extraction_smoke.py"],
    "benchmarking.rst": ["tests/smoke/test_benchmark_cli.py"],
    "optimization.rst": ["tests/smoke/test_optimize_cli.py"],
    "inference.rst": [
        "tests/unit/test_infer_cli.py",
        "tests/unit/test_inference_heatmaps.py",
    ],
    "model_packaging.rst": ["tests/unit/test_model_package.py"],
    "slide_retrieval.rst": ["tests/smoke/test_slide_retrieval_e2e_smoke.py"],
    "cli.rst": ["tests/unit/test_docs_cli_consistency.py"],
}


def _rst_code_blocks(text: str, language: str) -> list[str]:
    """Extract dedented bodies for one RST code-block language."""

    lines = text.splitlines()
    blocks: list[str] = []
    index = 0
    directive = re.compile(rf"^(\s*)\.\. code-block:: {language}\s*$")
    while index < len(lines):
        if directive.match(lines[index]) is None:
            index += 1
            continue
        cursor = index + 1
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            break
        indent = len(lines[cursor]) - len(lines[cursor].lstrip())
        body: list[str] = []
        while cursor < len(lines):
            line = lines[cursor]
            current_indent = len(line) - len(line.lstrip())
            if line.strip() and current_indent < indent:
                break
            body.append(line[indent:] if line.strip() else "")
            cursor += 1
        blocks.append("\n".join(body))
        index = cursor
    return blocks


def test_every_focused_tutorial_has_workflow_tests() -> None:
    """Every tutorial page is tied to an executable workflow regression test."""

    focused = {path.name for path in TUTORIAL_DIR.glob("*.rst")} - {"index.rst"}
    assert set(WORKFLOW_COVERAGE) == focused
    for test_paths in WORKFLOW_COVERAGE.values():
        for relative_path in test_paths:
            path = REPO_ROOT / relative_path
            assert path.is_file(), f"Missing tutorial workflow test: {relative_path}"
            assert "def test_" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("tutorial", sorted(WORKFLOW_COVERAGE))
def test_tutorial_python_blocks_compile(tutorial: str) -> None:
    """All Python snippets in tutorial pages must remain syntactically executable."""

    path = TUTORIAL_DIR / tutorial
    for index, source in enumerate(
        _rst_code_blocks(path.read_text(encoding="utf-8"), "python")
    ):
        compile(source, f"{path.name}:python-block-{index}", "exec")
