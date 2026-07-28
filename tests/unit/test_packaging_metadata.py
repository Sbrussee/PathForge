"""Regression tests for installable dependency metadata."""

from __future__ import annotations

import tomllib
from pathlib import Path


PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_tcga_tools_is_optional_and_uses_portable_git_source() -> None:
    """TCGA support must not require a sibling checkout of TCGA-Tools."""
    metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    runtime_dependencies = metadata["project"]["dependencies"]
    optional_dependencies = metadata["project"]["optional-dependencies"]
    tcga_source = metadata["tool"]["uv"]["sources"]["tcga-tools"]

    assert "tcga-tools" not in runtime_dependencies
    assert optional_dependencies["tcga"] == ["tcga-tools"]
    assert "tcga-tools" not in optional_dependencies["docs"]
    assert tcga_source["git"] == "https://github.com/LUMCPathAI/TCGA-Tools.git"
    assert tcga_source["rev"]
    assert "path" not in tcga_source
    assert "editable" not in tcga_source


def test_lazyslide_stack_is_installed_by_default() -> None:
    """The primary WSI feature-extraction backend must not require an extra."""
    metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    runtime_dependencies = metadata["project"]["dependencies"]
    optional_dependencies = metadata["project"]["optional-dependencies"]

    expected = {"lazyslide>=0.10.0", "wsidata", "timm", "geopandas", "anndata>=0.10.9"}
    assert expected <= set(runtime_dependencies)
    assert "lazyslide" not in optional_dependencies
