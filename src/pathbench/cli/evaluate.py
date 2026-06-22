from __future__ import annotations

import argparse

from pathbench.core.evaluation import EvaluationOrchestrator
from pathbench.cli.base import (
    add_config_argument,
    add_log_level_argument,
    configure_logging,
    load_experiment,
)


def main(argv: list[str] | None = None) -> int:
    """
    Run the evaluation CLI entrypoint.

    Inputs:
    - `argv`: optional command-line argument list.

    Outputs:
    - Process exit code. Returns `0` when evaluation completes successfully.

    Example:
    ```python
    exit_code = main(["--config", "configs/config.yaml"])
    ```
    """
    parser = argparse.ArgumentParser(description="Evaluation workflow")
    add_config_argument(parser)
    add_log_level_argument(parser)
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    experiment = load_experiment(args.config)
    orchestrator = EvaluationOrchestrator(experiment)
    orchestrator.evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
