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
from urllib.parse import unquote

import pytest

DOCS_DIR = Path(__file__).parent.parent.parent / "docs"
REPO_ROOT = DOCS_DIR.parent


# ---------------------------------------------------------------------------
# Test 0 — canonical product name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "path",
    [REPO_ROOT / "README.md", *DOCS_DIR.rglob("*.rst"), *DOCS_DIR.rglob("*.md")],
)
def test_legacy_pathforge_mil_branding_is_absent(path: Path) -> None:
    """Use the canonical ``PathForge`` product name in user-facing docs."""
    text = path.read_text(encoding="utf-8")
    legacy_name = "PathForge" + "-MIL"
    assert legacy_name not in text


def test_readme_local_links_exist() -> None:
    """Keep repository-relative documentation links in the README valid."""
    readme = REPO_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    link_targets = re.findall(r"(?<!!)\[[^]]+\]\(([^)]+)\)", text)
    missing: list[str] = []
    for raw_target in link_targets:
        target = raw_target.split("#", maxsplit=1)[0].strip()
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        resolved = REPO_ROOT / unquote(target)
        if not resolved.exists():
            missing.append(raw_target)
    assert not missing, f"README links point to missing local files: {missing}"


def test_readme_documentation_links_use_readthedocs() -> None:
    """Direct README readers to rendered documentation rather than sources."""
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "](docs/" not in text
    assert "https://pathforge.readthedocs.io/" in text


def test_mil_options_match_code_catalogs() -> None:
    """Keep the documented static option names aligned with code catalogs."""
    from pathforge.adapters.mil_lab.backend import MILLAB_MODEL_SPECS
    from pathforge.adapters.torchmil.backend import TORCHMIL_MODEL_SPECS
    from pathforge.config.config import BenchmarkParameters
    from pathforge.core.models.sklearn_slide import SLIDE_LEVEL_MODEL_NAMES
    from pathforge.utils.registries import LOSSES, list_mil_models

    text = (DOCS_DIR / "mil_options.rst").read_text(encoding="utf-8")
    mil_grid_fields = set(BenchmarkParameters.model_fields) - {
        "retrieval_representation",
        "search_strategy",
    }
    expected_names = {
        *mil_grid_fields,
        *MILLAB_MODEL_SPECS,
        *TORCHMIL_MODEL_SPECS,
        *SLIDE_LEVEL_MODEL_NAMES,
        *(item.name for item in list_mil_models()),
        *LOSSES.list_plugins(),
    }
    missing = sorted(name for name in expected_names if name not in text)
    assert not missing, f"MIL option names missing from documentation: {missing}"


def test_task_outputs_match_metric_and_visualization_catalogs() -> None:
    """Keep task-output options aligned with configuration and registries."""
    from pathforge.config.config import (
        CLASSIFICATION_METRIC_NAMES,
        REGRESSION_METRIC_NAMES,
        SURVIVAL_METRIC_NAMES,
    )
    from pathforge.core.evaluation.registry import (
        import_evaluation_metric_modules,
        list_evaluation_metrics,
    )
    from pathforge.slide_retrieval.visualization.service import (
        _SUPPORTED_VISUALIZATIONS,
    )

    import_evaluation_metric_modules()
    text = (DOCS_DIR / "task_outputs.rst").read_text(encoding="utf-8")
    expected_names = {
        *CLASSIFICATION_METRIC_NAMES,
        *REGRESSION_METRIC_NAMES,
        *SURVIVAL_METRIC_NAMES,
        *list_evaluation_metrics("slide_retrieval"),
        *_SUPPORTED_VISUALIZATIONS,
    }
    missing = sorted(name for name in expected_names if name not in text)
    assert not missing, f"Task output options missing from documentation: {missing}"


def test_pipeline_optimization_terminology() -> None:
    """Use the workflow-level Pipeline Optimization name consistently."""
    paths = [
        REPO_ROOT / "README.md",
        *DOCS_DIR.rglob("*.rst"),
        *DOCS_DIR.rglob("*.md"),
    ]
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in paths
        if "hyperparameter optimization" in path.read_text(encoding="utf-8").lower()
    ]
    assert not offenders, f"Legacy optimization terminology remains in: {offenders}"


def test_tutorials_select_concrete_mil_model_names() -> None:
    """Do not teach legacy backend sentinels as benchmark model choices."""
    paths = [REPO_ROOT / "README.md", *DOCS_DIR.joinpath("tutorials").rglob("*.rst")]
    pattern = re.compile(r"mil:\s*\[(?:torchmil|mil-lab)\]", re.IGNORECASE)
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in paths
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"Legacy MIL backend sentinels remain in tutorials: {offenders}"


def test_documented_tutorial_models_are_in_backend_catalog() -> None:
    """Keep inline tutorial model grids aligned with the public model catalog."""
    from pathforge.utils.registries import list_mil_models

    paths = [REPO_ROOT / "README.md", *DOCS_DIR.joinpath("tutorials").rglob("*.rst")]
    documented: set[str] = set()
    pattern = re.compile(r"^\s*mil:\s*\[([^]]*)\]", re.MULTILINE)
    for path in paths:
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            documented.update(
                name.strip() for name in match.group(1).split(",") if name.strip()
            )

    catalog = {entry.name for entry in list_mil_models()}
    unknown = sorted(documented - catalog)
    assert not unknown, f"Tutorials reference MIL models outside the catalog: {unknown}"


def test_end_to_end_tutorial_is_first() -> None:
    """Present the complete workflow before focused subtask tutorials."""
    text = (DOCS_DIR / "tutorials" / "index.rst").read_text(encoding="utf-8")
    assert text.index("   end_to_end") < text.index("   feature_extraction")


def test_introduction_leads_getting_started_and_covers_supported_tasks() -> None:
    """Introduce PathForge's purpose and tasks before setup instructions."""
    index_text = (DOCS_DIR / "index.rst").read_text(encoding="utf-8")
    assert index_text.index("   introduction") < index_text.index("   installation")

    introduction = (DOCS_DIR / "introduction.rst").read_text(encoding="utf-8")
    required_topics = {
        "Benchmarking",
        "Pipeline optimization",
        "Classification",
        "Regression",
        "Continuous survival",
        "Discrete survival",
        "Slide Retrieval",
    }
    missing = sorted(topic for topic in required_topics if topic not in introduction)
    assert not missing, f"Introduction is missing required topics: {missing}"


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


def _collect_autodoc_modules(docs_dir: Path) -> list[tuple[str, str, Path]]:
    """Return directive kind, target, and source for every autodoc directive."""
    pattern = re.compile(
        r"^\.\. auto(module|class|function|exception|data)::\s+(.+)$"
    )
    results: list[tuple[str, str, Path]] = []
    for rst_file in docs_dir.rglob("*.rst"):
        text = rst_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            m = pattern.match(line.strip())
            if m:
                kind, target = m.group(1), m.group(2).strip()
                results.append((kind, target, rst_file))
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
    for kind, target, src in items:
        # ``automodule`` targets are already modules. Other directives target
        # members, so import their parent module.
        module_candidate = (
            target if kind == "module" else target.rsplit(".", maxsplit=1)[0]
        )
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
    """Build the HTML documentation and verify that equations are rendered."""
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
    metrics_html = (
        build_dir / "slide-retrieval-results-and-metrics.html"
    ).read_text(encoding="utf-8")
    assert "mathjax" in metrics_html.lower()
    assert 'class="math notranslate nohighlight"' in metrics_html
    examples_html = (build_dir / "api" / "examples.html").read_text(
        encoding="utf-8"
    )
    assert "build_bag_id" in examples_html
    assert "pathforge benchmark run --help" in examples_html
