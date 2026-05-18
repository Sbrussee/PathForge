import logging
from pathlib import Path

import typer

from ..config.config import Config
from .common import LOG_LEVEL_CHOICES, configure_logging, enable_dask_query_planning
from ..core.experiments.base import Experiment
from ..policy.feature_extraction import FeatureExtractionPolicy


def run_feature_extraction(
    *,
    config: Path,
    log_level: str = "INFO",
) -> int:
    config_path = Path(config)
    configure_logging(log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting feature extraction CLI")
    logger.info("Using config: %s", config_path)

    enable_dask_query_planning()

    cfg = Config.from_yaml(config_path)
    experiment = Experiment(cfg)

    policy = FeatureExtractionPolicy(experiment)
    out = policy.execute()
    logger.info("Experiment finished with status: %s", out)
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
    raise SystemExit(run_feature_extraction(config=config, log_level=log_level))


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Feature extraction only")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVEL_CHOICES,
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args(argv)
    return run_feature_extraction(config=args.config, log_level=args.log_level)


if __name__ == "__main__":
    raise SystemExit(main())
