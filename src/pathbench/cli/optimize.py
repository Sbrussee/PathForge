from __future__ import annotations

import argparse

from ..policy.optimization import OptimizationPolicy
from .base import (
    add_config_argument,
    add_log_level_argument,
    configure_logging,
    load_config,
)


def main(argv: list[str] | None = None) -> int:
    """Run the PathBench optimization CLI."""
    parser = argparse.ArgumentParser(description="Run PathBench optimization")
    add_config_argument(parser)
    add_log_level_argument(parser)
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    cfg = load_config(args.config)
    OptimizationPolicy(cfg).execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
