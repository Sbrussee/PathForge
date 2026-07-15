from pathlib import Path


def test_readme_documents_optional_backends_and_run_modes():
    readme = Path("README.md").read_text(encoding="utf-8")

    required_phrases = [
        "TorchMIL",
        "MIL-Lab",
        "mil.backend",
        "mil.torchmil_model",
        "mil-lab",
        "benchmark",
        "optimization",
        "Integration Boundaries",
        "optional",
        "registry",
    ]

    for phrase in required_phrases:
        assert phrase in readme


def test_backend_docs_list_all_mil_backend_modes() -> None:
    """Keep the backend overview aligned with the config backend choices."""
    backend_docs = Path("docs/backends.rst").read_text(encoding="utf-8")
    for backend in ("native", "torchmil", "mil-lab"):
        assert backend in backend_docs

    assert "PathForge currently catalogs ``ABMIL``, ``DSMIL``, and ``CLAM``" in backend_docs
    assert "MIL-Lab is not installed by the ``mil-backends`` extra" in backend_docs
