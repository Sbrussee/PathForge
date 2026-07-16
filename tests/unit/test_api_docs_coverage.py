from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTERS_API_DOC = REPO_ROOT / "docs" / "api" / "adapters.rst"
API_DOCS = REPO_ROOT / "docs" / "api"

# Supported modules that form the user- or extension-facing API. Concrete
# retrieval strategy implementations are discovered through their documented
# registries and deliberately remain internal.
REQUIRED_PUBLIC_MODULES = {
    "pathforge.config.config",
    "pathforge.core.datasets.bag_dataset",
    "pathforge.core.datasets.bag_schema",
    "pathforge.core.evaluation.orchestrator",
    "pathforge.core.experiments.combo_ids",
    "pathforge.core.io.h5.descriptors",
    "pathforge.core.io.slide_artifacts.base",
    "pathforge.core.tasks.registry",
    "pathforge.execution.distributed",
    "pathforge.inference.model_package",
    "pathforge.policy.utils",
    "pathforge.slide_retrieval.representation_strategies.base",
    "pathforge.slide_retrieval.representation_strategies.registry",
    "pathforge.slide_retrieval.search_strategies.base",
    "pathforge.slide_retrieval.search_strategies.registry",
    "pathforge.training.lightning",
    "pathforge.utils.registries",
}


def test_adapters_api_docs_reference_loss_adapters() -> None:
    text = ADAPTERS_API_DOC.read_text(encoding="utf-8")

    assert ".. automodule:: pathforge.adapters.losses" in text


def test_adapters_api_docs_do_not_reference_deleted_bag_adapter() -> None:
    text = ADAPTERS_API_DOC.read_text(encoding="utf-8")

    assert "pathforge.adapters.torchmil.bag_adapter" not in text


def test_supported_public_modules_are_in_api_reference() -> None:
    """Keep supported modules visible when the source tree gains new APIs."""

    api_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(API_DOCS.glob("*.rst"))
    )
    documented_modules = set(
        re.findall(
            r"^\.\.\s+(?:automodule|autoclass|autofunction)::\s+"
            r"(pathforge(?:\.[A-Za-z_][A-Za-z0-9_]*)+)",
            api_text,
            flags=re.MULTILINE,
        )
    )

    missing = sorted(
        module
        for module in REQUIRED_PUBLIC_MODULES
        if not any(
            documented == module or documented.startswith(f"{module}.")
            for documented in documented_modules
        )
    )
    assert not missing, f"Public modules missing from API reference: {missing}"
