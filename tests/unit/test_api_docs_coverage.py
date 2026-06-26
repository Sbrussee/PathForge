from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTERS_API_DOC = REPO_ROOT / "docs" / "api" / "adapters.rst"


def test_adapters_api_docs_reference_loss_adapters() -> None:
    text = ADAPTERS_API_DOC.read_text(encoding="utf-8")

    assert ".. automodule:: pathforge.adapters.losses" in text


def test_adapters_api_docs_do_not_reference_deleted_bag_adapter() -> None:
    text = ADAPTERS_API_DOC.read_text(encoding="utf-8")

    assert "pathforge.adapters.torchmil.bag_adapter" not in text
