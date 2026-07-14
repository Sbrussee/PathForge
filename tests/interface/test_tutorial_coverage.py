"""Ensure every tutorial is backed by executable regression coverage."""

from __future__ import annotations

import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIALS_DIR = REPO_ROOT / "docs" / "tutorials"

# Each selector names a test collected by the normal pytest suite. Keeping this
# manifest explicit makes a newly added workflow tutorial fail until its real
# implementation path has smoke or integration coverage.
TUTORIAL_COVERAGE: dict[str, tuple[str, ...]] = {
    "benchmarking.rst": (
        "tests/unit/test_policy_benchmark.py::test_execute_combination_resolves_features_builds_datasets_and_runs_one_task",
    ),
    "cli.rst": (
        "tests/unit/test_docs_cli_consistency.py::test_documented_cli_options_exist",
    ),
    "end_to_end.rst": (
        "tests/smoke/test_feature_extract_cli.py::test_feature_extraction_thumbnail_write_smoke",
        "tests/unit/test_policy_benchmark.py::test_execute_combination_resolves_features_builds_datasets_and_runs_one_task",
        "tests/unit/test_model_package.py::test_inference_cli_reads_packaged_model_and_h5",
    ),
    "feature_extraction.rst": (
        "tests/smoke/test_feature_extract_cli.py::test_feature_extraction_thumbnail_write_smoke",
    ),
    "inference.rst": (
        "tests/unit/test_model_package.py::test_inference_cli_reads_packaged_model_and_h5",
        "tests/unit/test_inference_heatmaps.py::test_create_inference_heatmap_persists_h5_and_json_sidecar",
    ),
    "model_packaging.rst": (
        "tests/unit/test_model_package.py::test_packaged_model_roundtrip_preserves_predictions",
    ),
    "optimization.rst": (
        "tests/unit/test_policy_optimization.py::test_optimization_execute_writes_summary_csv_and_visualizations",
    ),
    "slide_retrieval.rst": (
        "tests/integration/test_slide_retrieval.py::test_reference_query_split_produces_results",
    ),
}


def _top_level_test_names(path: Path) -> set[str]:
    """Return top-level pytest function names declared in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def _rst_code_blocks(path: Path, language: str) -> list[str]:
    """Extract code bodies for one RST ``code-block`` language."""
    lines = path.read_text(encoding="utf-8").splitlines()
    directive = re.compile(rf"^(\s*)\.\. code-block:: {re.escape(language)}\s*$")
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        if not directive.match(lines[index]):
            index += 1
            continue
        cursor = index + 1
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            break
        indent = len(lines[cursor]) - len(lines[cursor].lstrip())
        body: list[str] = []
        while cursor < len(lines) and (
            not lines[cursor].strip()
            or len(lines[cursor]) - len(lines[cursor].lstrip()) >= indent
        ):
            body.append(lines[cursor][indent:] if lines[cursor].strip() else "")
            cursor += 1
        blocks.append("\n".join(body))
        index = cursor
    return blocks


def test_every_workflow_tutorial_has_executable_coverage() -> None:
    """Every workflow tutorial must map to one or more collected tests."""
    tutorial_names = {
        path.name for path in TUTORIALS_DIR.glob("*.rst") if path.name != "index.rst"
    }
    assert set(TUTORIAL_COVERAGE) == tutorial_names

    missing: list[str] = []
    for tutorial, selectors in TUTORIAL_COVERAGE.items():
        assert selectors, f"{tutorial} does not name any executable tests"
        for selector in selectors:
            relative_path, test_name = selector.split("::", maxsplit=1)
            test_path = REPO_ROOT / relative_path
            if not test_path.is_file() or test_name not in _top_level_test_names(test_path):
                missing.append(f"{tutorial} -> {selector}")
    assert not missing, f"Tutorial coverage points to missing tests: {missing}"


def test_tutorial_python_blocks_compile() -> None:
    """All Python snippets in tutorials must remain syntactically valid."""
    failures: dict[str, str] = {}
    for tutorial in sorted(TUTORIALS_DIR.glob("*.rst")):
        for index, block in enumerate(_rst_code_blocks(tutorial, "python"), start=1):
            try:
                compile(block, f"{tutorial.name}:python-block-{index}", "exec")
            except SyntaxError as error:
                failures[f"{tutorial.name} block {index}"] = str(error)
    assert not failures, f"Tutorial Python blocks do not compile: {failures}"
