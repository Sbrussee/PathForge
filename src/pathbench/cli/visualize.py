from __future__ import annotations

import argparse

from pathbench.cli.base import (
    add_config_argument,
    add_log_level_argument,
    configure_logging,
    load_experiment,
)
from pathbench.core.visualization import VisualizationOrchestrator


def main(argv: list[str] | None = None) -> int:
    """Run the visualization CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Visualization workflow")
    add_config_argument(parser)
    add_log_level_argument(parser)
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    experiment = load_experiment(args.config)
    orchestrator = VisualizationOrchestrator(experiment)
    orchestrator.visualize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
