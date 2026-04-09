from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pathbench.config.config import Config
from pathbench.core.evaluation import EvaluationOrchestrator
from pathbench.core.experiments.base import Experiment


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

    cfg = Config.from_yaml(args.config)
    experiment = Experiment(cfg)
    orchestrator = EvaluationOrchestrator(experiment)
    orchestrator.evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
