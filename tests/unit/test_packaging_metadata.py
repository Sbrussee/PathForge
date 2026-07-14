"""Regression tests for installable dependency metadata."""

from __future__ import annotations

import tomllib
from pathlib import Path


PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_tcga_tools_is_optional_for_pypi_and_docs_installs() -> None:
    """Read the Docs must not resolve the repository-local TCGA dependency."""
    metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    runtime_dependencies = metadata["project"]["dependencies"]
    optional_dependencies = metadata["project"]["optional-dependencies"]

    assert "tcga-tools" not in runtime_dependencies
    assert optional_dependencies["tcga"] == ["tcga-tools"]
    assert "tcga-tools" not in optional_dependencies["docs"]


def test_lazyslide_stack_is_installed_by_default() -> None:
    """The primary WSI feature-extraction backend must not require an extra."""
    metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    runtime_dependencies = metadata["project"]["dependencies"]
    optional_dependencies = metadata["project"]["optional-dependencies"]

    expected = {"lazyslide>=0.10.0", "wsidata", "timm", "geopandas", "anndata>=0.10.9"}
    assert expected <= set(runtime_dependencies)
    assert "lazyslide" not in optional_dependencies
