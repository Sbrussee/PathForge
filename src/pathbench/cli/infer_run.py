from __future__ import annotations

import logging
from pathlib import Path

import typer

from ..config.config import Config
from ..core.experiments.base import Experiment
from ..policy.inference import InferencePolicy
from .common import LOG_LEVEL_CHOICES, configure_logging, enable_dask_query_planning


def run_inference(
    *,
    config: Path,
    input_csv: Path,
    log_level: str = "INFO",
) -> int:
    config_path = Path(config)
    input_csv_path = Path(input_csv)
    configure_logging(log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting inference CLI")
    logger.info("Using config: %s", config_path)
    logger.info("Using inference input CSV: %s", input_csv_path)

    enable_dask_query_planning()

    cfg = Config.from_yaml(config_path)
    if cfg.experiment.mode != "inference":
        raise ValueError(
            "Inference CLI requires experiment.mode='inference'. "
            f"Got {cfg.experiment.mode!r}."
        )

    experiment = Experiment(cfg)
    policy = InferencePolicy(experiment)
    output = policy.execute(input_csv=input_csv_path)
    logger.info("Inference finished with status: %s", output)
    return 0


def run_command(
    config: Path = typer.Option(..., "--config", help="Path to YAML config."),
    input_csv: Path = typer.Option(
        ...,
        "--input-csv",
        help="CSV selecting slides to run inference for.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level.",
        show_default=True,
    ),
) -> None:
    raise SystemExit(
        run_inference(
            config=config,
            input_csv=input_csv,
            log_level=log_level,
        )
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Inference workflow")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument(
        "--input-csv",
        required=True,
        type=Path,
        help="CSV selecting slides to run inference for.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVEL_CHOICES,
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args(argv)
    return run_inference(
        config=args.config,
        input_csv=args.input_csv,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    raise SystemExit(main())
