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
import io
import re
import shlex
import tomllib
from contextlib import redirect_stderr, redirect_stdout
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
DOCUMENTED_COMMAND_RE = re.compile(
    r"(?m)^\s*(?:uv run )?(pathforge(?:-[a-z0-9]+)*(?:\s+[^\n|;]+)?)$"
)


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


def _documented_pathforge_commands(text: str) -> list[list[str]]:
    """Return tokenized PathForge commands from shell examples."""
    normalized = re.sub(r"\\\s*\n\s*", " ", text)
    commands: list[list[str]] = []
    for match in DOCUMENTED_COMMAND_RE.finditer(normalized):
        try:
            tokens = shlex.split(match.group(1))
        except ValueError:
            continue
        if tokens and (tokens[0] == "pathforge" or tokens[0].startswith("pathforge-")):
            commands.append(tokens)
    return commands


def _console_script_help(script: str) -> str:
    """Render ``--help`` for one declared argparse console script."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    target = data["project"]["scripts"][script]
    module_name, callable_name = target.split(":", maxsplit=1)
    callback = getattr(importlib.import_module(module_name), callable_name)
    output = io.StringIO()
    with redirect_stdout(output), redirect_stderr(output):
        with pytest.raises(SystemExit) as exit_info:
            callback(["--help"])
    assert exit_info.value.code == 0
    return output.getvalue()


def _typer_help(tokens: list[str]) -> str:
    """Render help for one documented umbrella-command path."""
    from typer.testing import CliRunner

    from pathforge.cli.app import app

    command_path = tokens[1:3]
    result = CliRunner().invoke(app, [*command_path, "--help"])
    assert result.exit_code == 0, result.output
    return result.output


def test_documented_cli_options_exist() -> None:
    """Every documented PathForge option must exist on the shown command."""
    declared = _declared_console_scripts()
    help_cache: dict[tuple[str, ...], str] = {}
    offenders: dict[str, list[str]] = {}

    for path, text in _doc_texts().items():
        for tokens in _documented_pathforge_commands(text):
            script = tokens[0]
            if script == "pathforge":
                if len(tokens) < 3:
                    continue
                key = tuple(tokens[:3])
                help_text = help_cache.setdefault(key, _typer_help(tokens))
                argument_tokens = tokens[3:]
            elif script in declared:
                key = (script,)
                help_text = help_cache.setdefault(key, _console_script_help(script))
                argument_tokens = tokens[1:]
            else:
                continue

            unknown = sorted(
                {
                    token.split("=", maxsplit=1)[0]
                    for token in argument_tokens
                    if token.startswith("--")
                    and token.split("=", maxsplit=1)[0] not in help_text
                }
            )
            if unknown:
                label = f"{path.relative_to(REPO_ROOT)} :: {' '.join(key)}"
                offenders.setdefault(label, []).extend(unknown)

    assert not offenders, f"Docs use options not exposed by the shown command: {offenders}"


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
