from __future__ import annotations

import logging
from pathlib import Path

import typer

from pathbench.config.config import Config
from pathbench.cli.common import LOG_LEVEL_CHOICES, configure_logging
from pathbench.core.experiments.base import Experiment
from pathbench.core.visualization import VisualizationOrchestrator

logger = logging.getLogger(__name__)


def run_visualization(
    *,
    config: Path,
    log_level: str = "INFO",
) -> int:
    """Run the visualization orchestrator for one YAML config and return an exit code."""
    config_path = Path(config)
    configure_logging(log_level)
    cfg = Config.from_yaml(config_path)
    experiment = Experiment(cfg)
    orchestrator = VisualizationOrchestrator(experiment)
    summary = orchestrator.visualize()
    logger.info("Visualization finished with status: %s", summary.get("status"))
    logger.info("Discovered/processed runs: %s", summary.get("num_runs", 0))
    created_files = list(summary.get("created_files", []))
    logger.info("Created visualization files: %d", len(created_files))
    for path in created_files[:25]:
        logger.info("Created: %s", path)
    if len(created_files) > 25:
        logger.info("Created: ... %d more file(s)", len(created_files) - 25)
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
    """Typer command that runs visualization from the provided config option."""
    raise SystemExit(run_visualization(config=config, log_level=log_level))


def main(argv: list[str] | None = None) -> int:
    """Argparse entry point for the visualization CLI; returns a process exit code."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualization workflow")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVEL_CHOICES,
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args(argv)
    return run_visualization(config=args.config, log_level=args.log_level)


if __name__ == "__main__":
    raise SystemExit(main())
