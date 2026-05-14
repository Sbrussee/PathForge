from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..config.config import Config
from ..policy.benchmarking import BenchmarkingPolicy


def main(argv: list[str] | None = None) -> int:
    """Run the PathBench benchmarking CLI."""
    parser = argparse.ArgumentParser(description="Run PathBench benchmarking")
    parser.add_argument("--config", required=True, help="Path to YAML config")
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

    cfg = Config.from_yaml(Path(args.config))
    BenchmarkingPolicy(cfg).execute()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
