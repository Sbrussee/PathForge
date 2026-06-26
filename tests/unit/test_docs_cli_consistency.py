"""Guardrail tests: every CLI command shown in the README and docs must be real.

These tests parse the documentation for the three ways the CLI is invoked and
verify each against the actual project:

* console scripts ``pathforge-...`` must be declared in ``[project.scripts]``;
* ``python -m pathforge.cli.<module>`` targets must be importable;
* ``pathforge <group> <command>`` invocations must exist in the Typer app.

They fail loudly when documentation drifts from the code, so the instructions in
the README/docs stay correct.
"""

from __future__ import annotations

import importlib
import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

DOC_FILES = [
    path
    for path in (
        [REPO_ROOT / "README.md"]
        + sorted((REPO_ROOT / "docs").rglob("*.rst"))
        + sorted((REPO_ROOT / "docs").rglob("*.md"))
    )
    if path.is_file()
]

CONSOLE_SCRIPT_RE = re.compile(r"\bpathforge(?:-[a-z0-9]+)+\b")
PYTHON_M_RE = re.compile(r"python -m (pathforge\.cli\.[a-z_]+)")
TYPER_INVOCATION_RE = re.compile(r"\bpathforge ([a-z-]+) ([a-z][a-z-]*)\b")


def _declared_console_scripts() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return set(data["project"]["scripts"])


def _typer_command_map() -> dict[str, set[str]]:
    from pathforge.cli.app import app

    mapping: dict[str, set[str]] = {}
    for group in app.registered_groups:
        sub = group.typer_instance
        mapping[group.name] = {
            command.name or command.callback.__name__
            for command in sub.registered_commands
        }
    return mapping


def _doc_texts() -> dict[Path, str]:
    return {path: path.read_text(encoding="utf-8") for path in DOC_FILES}


def test_doc_files_exist() -> None:
    assert DOC_FILES, "Expected README.md and docs/ files to scan"


def test_documented_console_scripts_are_declared() -> None:
    declared = _declared_console_scripts()
    offenders: dict[str, set[str]] = {}
    for path, text in _doc_texts().items():
        referenced = set(CONSOLE_SCRIPT_RE.findall(text)) - {"pathforge"}
        unknown = {name for name in referenced if name not in declared}
        if unknown:
            offenders[str(path.relative_to(REPO_ROOT))] = unknown
    assert not offenders, (
        "Docs reference console scripts not declared in [project.scripts]: "
        f"{offenders}. Declared: {sorted(declared)}"
    )


def test_documented_python_m_modules_are_importable() -> None:
    offenders: dict[str, dict[str, str]] = {}
    for path, text in _doc_texts().items():
        for module_name in set(PYTHON_M_RE.findall(text)):
            try:
                importlib.import_module(module_name)
            except Exception as error:  # noqa: BLE001 - report any import failure
                offenders.setdefault(str(path.relative_to(REPO_ROOT)), {})[
                    module_name
                ] = repr(error)
    assert not offenders, f"Docs reference non-importable CLI modules: {offenders}"


def test_documented_typer_commands_exist() -> None:
    command_map = _typer_command_map()
    offenders: dict[str, set[str]] = {}
    for path, text in _doc_texts().items():
        for group, command in set(TYPER_INVOCATION_RE.findall(text)):
            if command not in command_map.get(group, set()):
                offenders.setdefault(str(path.relative_to(REPO_ROOT)), set()).add(
                    f"pathforge {group} {command}"
                )
    assert not offenders, (
        f"Docs reference unknown 'pathforge <group> <command>' invocations: {offenders}. "
        f"Valid: { {g: sorted(c) for g, c in command_map.items()} }"
    )


def _yaml_code_blocks(text: str) -> list[str]:
    """Extract YAML code blocks from Markdown fences and RST ``code-block`` directives."""
    blocks: list[str] = []
    for match in re.finditer(r"```ya?ml\n(.*?)```", text, re.S):
        blocks.append(match.group(1))

    lines = text.splitlines()
    index = 0
    directive = re.compile(r"^(\s*)\.\. code-block:: ya?ml\s*$")
    while index < len(lines):
        match = directive.match(lines[index])
        if not match:
            index += 1
            continue
        cursor = index + 1
        while cursor < len(lines) and lines[cursor].strip() == "":
            cursor += 1
        if cursor >= len(lines):
            break
        indent = len(lines[cursor]) - len(lines[cursor].lstrip())
        body: list[str] = []
        while cursor < len(lines) and (
            lines[cursor].strip() == ""
            or len(lines[cursor]) - len(lines[cursor].lstrip()) >= indent
        ):
            body.append(lines[cursor][indent:] if lines[cursor].strip() else "")
            cursor += 1
        blocks.append("\n".join(body))
        index = cursor
    return blocks


def test_documented_full_yaml_configs_validate() -> None:
    """Complete config examples in the docs must validate against the schema."""
    import yaml

    from pathforge.config.config import Config

    offenders: dict[str, str] = {}
    for path, text in _doc_texts().items():
        for raw in _yaml_code_blocks(text):
            if "project_name:" not in raw or "datasets:" not in raw:
                continue  # skip partial fragments
            try:
                data = yaml.safe_load(raw)
                Config.model_validate(data)
            except Exception as error:  # noqa: BLE001 - report any schema failure
                snippet = raw.strip().splitlines()[0] if raw.strip() else "<empty>"
                offenders[f"{path.relative_to(REPO_ROOT)} :: {snippet}"] = repr(error)
    assert not offenders, f"Docs contain invalid config examples: {offenders}"
