from pathlib import Path


def test_readme_documents_optional_backends_and_run_modes():
    readme = Path("README.md").read_text(encoding="utf-8")

    required_phrases = [
        "TorchMIL",
        "mil.backend",
        "mil.torchmil_model",
        "benchmark",
        "optimization",
        "Clean Architecture",
        "optional",
        "registry",
    ]

    for phrase in required_phrases:
        assert phrase in readme
