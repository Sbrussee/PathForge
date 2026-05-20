"""Tests that keep the user-facing testing guide aligned with the real suite."""

from __future__ import annotations

import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_TESTING = REPO_ROOT / "docs" / "testing.rst"
TESTING_MD = REPO_ROOT / "tests" / "TESTING.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_testing_pages_cover_all_test_layers() -> None:
    docs_text = _read(DOCS_TESTING)
    md_text = _read(TESTING_MD)

    for path_text in (
        "tests/unit",
        "tests/interface",
        "tests/integration",
        "tests/smoke",
    ):
        assert path_text in docs_text
        assert path_text in md_text

    expected = (
        "uv run pytest -q tests/unit tests/interface tests/integration tests/smoke"
    )
    assert expected in docs_text
    assert expected in md_text


def test_testing_pages_describe_pytest_outputs() -> None:
    text = _read(DOCS_TESTING).lower()
    for snippet in (
        "collected n items",
        "passed",
        "skipped",
        "failed",
        "traceback",
        "artifacts",
        "pathbench_smoke_cache",
    ):
        assert snippet in text


def test_smoke_command_matches_registered_marker() -> None:
    pyproject = tomllib.loads(_read(PYPROJECT))
    markers = pyproject["tool"]["pytest"]["ini_options"]["markers"]
    assert any(marker.startswith("smoke:") for marker in markers)

    text = _read(DOCS_TESTING)
    assert "pytest -q -m smoke tests/smoke" in text
