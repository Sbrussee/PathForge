# src/pathbench/cli/feature_extract.py

import argparse
from pathlib import Path
import logging

import dask

from ..config.config import Config
from ..core.experiments.base import Experiment
from ..policy.feature_extraction import FeatureExtractionPolicy


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Feature extraction only")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level (default: INFO)",)
    args = parser.parse_args(argv)

    # ---- logging config (once) ----
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting feature extraction CLI")
    logger.info(f"Using config: {args.config}")

    dask.config.set({"dataframe.query-planning": True})
    
    cfg = Config.from_yaml(Path(args.config))
    experiment = Experiment(cfg)

    policy = FeatureExtractionPolicy(experiment)
    out = policy.execute()
    logger.info(f"Experiment finished with status: {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())