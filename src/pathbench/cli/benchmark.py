from __future__ import annotations

import argparse
import logging
from pathlib import Path

import dask

from ..config.config import Config
from ..core.experiments.base import Experiment
from ..policy.benchmarking import BenchmarkingPolicy


def main(argv: list[str] | None = None) -> int:
    """
    Run the benchmarking CLI entrypoint.

    Inputs:
    - `argv`: `list[str] | None`
      Optional command-line arguments with shape `(n_args,)`. When `None`,
      arguments are read from `sys.argv`.

    Outputs:
    - `int`
      Process exit code. Returns `0` when the benchmarking policy completes
      successfully.

    Example:
    ```python
    exit_code = main(["--config", "configs/benchmark.yaml"])
    ```
    """
    parser = argparse.ArgumentParser(description="Benchmarking workflow")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting benchmarking CLI")
    logger.info("Using config: %s", args.config)

    dask.config.set({"dataframe.query-planning": True})

    cfg = Config.from_yaml(args.config)
    experiment = Experiment(cfg)
    policy = BenchmarkingPolicy(experiment)

    output = policy.execute()
    logger.info("Benchmarking finished with status: %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
