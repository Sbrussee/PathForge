from __future__ import annotations

import argparse
import logging
from ..policy.benchmarking import BenchmarkingPolicy
from .base import (
    add_config_argument,
    add_log_level_argument,
    configure_logging,
    enable_dask_query_planning,
    load_experiment,
)


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
    add_config_argument(parser)
    add_log_level_argument(parser)
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting benchmarking CLI")
    logger.info("Using config: %s", args.config)

    enable_dask_query_planning()

    policy = BenchmarkingPolicy(load_experiment(args.config))

    output = policy.execute()
    logger.info("Benchmarking finished with status: %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
