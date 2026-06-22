from __future__ import annotations

import argparse
import logging

from ..policy.feature_extraction import FeatureExtractionPolicy
from .base import (
    add_config_argument,
    add_log_level_argument,
    configure_logging,
    enable_dask_query_planning,
    load_experiment,
)


def main(argv: list[str] | None = None) -> int:
    """Run the batch feature-extraction command-line entrypoint."""
    parser = argparse.ArgumentParser(description="Feature extraction only")
    add_config_argument(parser)
    add_log_level_argument(parser)
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting feature extraction CLI")
    logger.info("Using config: %s", args.config)

    enable_dask_query_planning()
    experiment = load_experiment(args.config)

    policy = FeatureExtractionPolicy(experiment)
    out = policy.execute()
    logger.info("Experiment finished with status: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
