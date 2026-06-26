from __future__ import annotations

import argparse
import logging
from pathlib import Path

import dask

from pathbench.config.config import Config
from pathbench.core.experiments.base import Experiment

LOG_LEVEL_CHOICES: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Register the canonical ``--config`` CLI argument on a parser."""
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")


def add_log_level_argument(parser: argparse.ArgumentParser) -> None:
    """Register the canonical ``--log-level`` CLI argument on a parser."""
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVEL_CHOICES,
        help="Logging level (default: INFO)",
    )


def configure_logging(log_level: str) -> None:
    """Configure the shared PathBench CLI logging format once per process."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=LOG_FORMAT,
    )


def enable_dask_query_planning() -> None:
    """Enable the shared Dask dataframe planning setting used by PathBench CLIs."""
    dask.config.set({"dataframe.query-planning": True})


def load_config(config_path: str | Path) -> Config:
    """Load a validated PathBench config from disk."""
    return Config.from_yaml(Path(config_path))


def load_experiment(config_path: str | Path) -> Experiment:
    """Build an :class:`Experiment` from one validated config path."""
    return Experiment(load_config(config_path))


__all__ = [
    "LOG_LEVEL_CHOICES",
    "LOG_FORMAT",
    "add_config_argument",
    "add_log_level_argument",
    "configure_logging",
    "enable_dask_query_planning",
    "load_config",
    "load_experiment",
]
