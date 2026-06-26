from __future__ import annotations

import logging
from pathlib import Path

import dask

LOG_LEVEL_CHOICES = ["DEBUG", "INFO", "WARNING", "ERROR"]
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def normalize_log_level(log_level: str) -> str:
    """Return the upper-cased log level, validating it against the allowed set."""
    normalized = str(log_level).upper()
    if normalized not in LOG_LEVEL_CHOICES:
        allowed = ", ".join(LOG_LEVEL_CHOICES)
        raise ValueError(f"Unsupported log level '{log_level}'. Expected one of: {allowed}")
    return normalized


def configure_logging(log_level: str) -> None:
    """Configure root logging at the given level using the shared CLI format."""
    normalized = normalize_log_level(log_level)
    logging.basicConfig(
        level=getattr(logging, normalized),
        format=LOG_FORMAT,
    )


def enable_dask_query_planning() -> None:
    """Enable Dask dataframe query planning for downstream processing."""
    dask.config.set({"dataframe.query-planning": True})


def resolve_config_path(config: Path) -> Path:
    """Return the config path as a :class:`Path`, raising if it does not exist."""
    config_path = Path(config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return config_path
