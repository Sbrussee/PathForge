# src/pathbench/cli/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class CLIBase(ABC):
    """
    Abstract CLI interface – concrete implementations can use argparse,
    click, typer, etc.
    """

    @abstractmethod
    def main(self, argv: list[str] | None = None) -> int:
        ...
