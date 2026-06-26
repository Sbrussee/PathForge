from __future__ import annotations

import importlib
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def test_console_scripts_target_importable_main_callables() -> None:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]

    for script_name, entrypoint in scripts.items():
        module_name, callable_name = entrypoint.split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        target = getattr(module, callable_name, None)
        assert callable(target), (
            f"Console script '{script_name}' points to '{entrypoint}', "
            "but that target is not importable as a callable."
        )
