from __future__ import annotations

import argparse
import logging
from pathlib import Path

import typer

from ..config.config import Config
from .common import LOG_LEVEL_CHOICES, configure_logging, enable_dask_query_planning
from ..core.experiments.base import Experiment
from ..policy.benchmarking import BenchmarkingPolicy


def run_benchmark(
    *,
    config: Path,
    log_level: str = "INFO",
) -> int:
    """Run the benchmarking workflow for one YAML config and return an exit code."""
    config_path = Path(config)
    configure_logging(log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting benchmarking CLI")
    logger.info("Using config: %s", config_path)

    enable_dask_query_planning()

    cfg = Config.from_yaml(config_path)
    experiment = Experiment(cfg)
    policy = BenchmarkingPolicy(experiment)

    output = policy.execute()
    logger.info("Benchmarking finished with status: %s", output)
    return 0


def run_command(
    config: Path = typer.Option(..., "--config", help="Path to YAML config"),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level.",
        show_default=True,
    ),
) -> None:
    """Typer command that runs benchmarking from the provided config option."""
    raise SystemExit(run_benchmark(config=config, log_level=log_level))


def main(argv: list[str] | None = None) -> int:
    """Argparse entry point for the benchmarking CLI; returns a process exit code."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmarking workflow")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVEL_CHOICES,
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args(argv)
    return run_benchmark(config=args.config, log_level=args.log_level)


if __name__ == "__main__":
    raise SystemExit(main())
