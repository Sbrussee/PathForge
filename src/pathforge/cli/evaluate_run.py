from __future__ import annotations

import logging
from pathlib import Path

import typer

from pathforge.config.config import Config
from pathforge.core.evaluation import EvaluationOrchestrator
from pathforge.core.experiments.base import Experiment
from pathforge.cli.common import LOG_LEVEL_CHOICES, configure_logging


def run_evaluation(
    *,
    config: Path,
    log_level: str = "INFO",
) -> int:
    """Run the evaluation orchestrator for one YAML config and return an exit code."""
    config_path = Path(config)
    configure_logging(log_level)
    cfg = Config.from_yaml(config_path)
    experiment = Experiment(cfg)
    orchestrator = EvaluationOrchestrator(experiment)
    orchestrator.evaluate()
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
    """Typer command that runs evaluation from the provided config option."""
    raise SystemExit(run_evaluation(config=config, log_level=log_level))


def main(argv: list[str] | None = None) -> int:
    """Argparse entry point for the evaluation CLI; returns a process exit code."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluation workflow")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVEL_CHOICES,
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args(argv)
    return run_evaluation(config=args.config, log_level=args.log_level)


if __name__ == "__main__":
    raise SystemExit(main())
