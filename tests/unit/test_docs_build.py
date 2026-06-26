# tests/unit/test_docs_build.py
"""
Structural tests for the Sphinx documentation.

These tests verify:
  1. Every RST file referenced in toctrees exists on disk.
  2. Every automodule/autoclass target module is importable.
  3. The Sphinx build completes without errors (skipped if sphinx is absent).
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).parent.parent.parent / "docs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_toctree_entries(rst_path: Path) -> list[tuple[Path, str]]:
    """Return (resolved_path, raw_entry) for every toctree entry in rst_path."""
    text = rst_path.read_text(encoding="utf-8")
    results: list[tuple[Path, str]] = []
    in_toctree = False
    base = rst_path.parent

    for line in text.splitlines():
        stripped = line.strip()
        if ".. toctree::" in stripped:
            in_toctree = True
            continue
        if in_toctree:
            if stripped.startswith("..") or (stripped and stripped.startswith(":")):
                continue
            if stripped == "":
                continue
            # A non-empty, non-directive line inside a toctree is an entry
            if stripped and not stripped.startswith(":"):
                entry = stripped
                resolved = (base / entry).with_suffix(".rst")
                results.append((resolved, entry))
                in_toctree = False  # toctrees end at blank lines; reset and re-scan
    return results


def _all_toctree_references(start: Path) -> list[tuple[Path, str, Path]]:
    """Recursively collect (resolved_path, entry, source_rst) from all toctrees."""
    visited: set[Path] = set()
    queue: list[Path] = [start]
    results: list[tuple[Path, str, Path]] = []

    while queue:
        current = queue.pop()
        if current in visited or not current.exists():
            continue
        visited.add(current)

        text = current.read_text(encoding="utf-8")
        # Collect all entries from toctrees in this file
        in_toctree = False
        for line in text.splitlines():
            stripped = line.strip()
            if ".. toctree::" in stripped:
                in_toctree = True
                continue
            if in_toctree:
                if not stripped:
                    in_toctree = False
                    continue
                if stripped.startswith(":") or stripped.startswith(".."):
                    continue
                entry = stripped
                resolved = (current.parent / entry).with_suffix(".rst")
                results.append((resolved, entry, current))
                queue.append(resolved)

    return results


def _collect_autodoc_modules(docs_dir: Path) -> list[tuple[str, Path]]:
    """Return (module_or_class_path, source_rst) for every autodoc directive."""
    pattern = re.compile(
        r"^\.\. auto(?:module|class|function|exception|data)::\s+(.+)$"
    )
    results: list[tuple[str, Path]] = []
    for rst_file in docs_dir.rglob("*.rst"):
        text = rst_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            m = pattern.match(line.strip())
            if m:
                target = m.group(1).strip()
                results.append((target, rst_file))
    return results


# ---------------------------------------------------------------------------
# Test 1 — all toctree references resolve to existing RST files
# ---------------------------------------------------------------------------

def _toctree_params():
    index = DOCS_DIR / "index.rst"
    if not index.exists():
        return []
    refs = _all_toctree_references(index)
    return [(str(resolved), entry, str(src)) for resolved, entry, src in refs]


@pytest.mark.parametrize("resolved_str,entry,src_str", _toctree_params())
def test_toctree_reference_exists(resolved_str: str, entry: str, src_str: str) -> None:
    resolved = Path(resolved_str)
    assert resolved.exists(), (
        f"toctree entry '{entry}' in '{src_str}' "
        f"resolves to '{resolved}' which does not exist"
    )


# ---------------------------------------------------------------------------
# Test 2 — all automodule/autoclass targets are importable
# ---------------------------------------------------------------------------

_SKIP_MODULES = frozenset(
    {
        # Optional extras not installed in the base environment
        "pathforge.core.models.mamba_mil",
        "pathforge.adapters.metrics.classification",   # requires torchmetrics
        "pathforge.adapters.metrics.survival",         # requires torchsurv
        "pathforge.adapters.tcga_tools",               # requires tcga-tools
        "pathforge.core.slide_processing.lazyslide",   # lazyslide-safe to import but may need GPU
    }
)


def _autodoc_params():
    if not DOCS_DIR.exists():
        return []
    items = _collect_autodoc_modules(DOCS_DIR)
    # Resolve class/function paths to their parent module
    results = []
    for target, src in items:
        # For autoclass/autofunction the target may be "package.module.ClassName"
        # We import only the module portion (everything up to last segment that starts with uppercase or is a known function)
        parts = target.split(".")
        # Try to find the longest importable prefix
        module_candidate = target
        if len(parts) > 1 and (parts[-1][0].isupper() or parts[-1][0].islower()):
            # Could be Module.Class or module.function — try the module prefix
            module_candidate = ".".join(parts[:-1])
        results.append((module_candidate, target, str(src)))
    # Deduplicate by module_candidate
    seen: set[str] = set()
    unique = []
    for mc, target, src in results:
        if mc not in seen:
            seen.add(mc)
            unique.append((mc, target, src))
    return unique


@pytest.mark.parametrize("module_path,target,src_str", _autodoc_params())
def test_autodoc_module_is_importable(
    module_path: str, target: str, src_str: str
) -> None:
    if module_path in _SKIP_MODULES:
        pytest.skip(f"optional dependency not installed: {module_path}")
    try:
        importlib.import_module(module_path)
    except ImportError as exc:
        pytest.fail(
            f"autodoc target '{target}' in '{src_str}' "
            f"requires module '{module_path}' which is not importable: {exc}"
        )


# ---------------------------------------------------------------------------
# Test 3 — Sphinx build completes without errors
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    importlib.util.find_spec("sphinx") is None,
    reason="sphinx not installed",
)
def test_sphinx_build_no_errors(tmp_path: Path) -> None:
    """Run a dummy Sphinx build and assert exit code 0."""
    import subprocess

    build_dir = tmp_path / "_build" / "html"
    result = subprocess.run(
        [
            sys.executable, "-m", "sphinx",
            "-b", "html",
            "-W",          # treat warnings as errors
            "--keep-going",
            str(DOCS_DIR),
            str(build_dir),
        ],
        capture_output=True,
        text=True,
            timeout=300,
        )
    assert result.returncode == 0, (
        f"Sphinx build failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
