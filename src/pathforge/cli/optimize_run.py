from pathlib import Path

import typer

from ..config.config import Config
from ..policy.optimization import OptimizationPolicy


def run_optimization(
    *,
    config: Path,
) -> int:
    """Run the optimization policy for one YAML config and return an exit code."""
    cfg = Config.from_yaml(config)
    OptimizationPolicy(cfg).execute()
    return 0


def run_command(
    config: Path = typer.Option(..., "--config", help="Path to YAML config."),
) -> None:
    """Typer command that runs optimization from the provided config option."""
    raise SystemExit(run_optimization(config=config))


def main(argv: list[str] | None = None) -> int:
    """Argparse entry point for the optimization CLI; returns a process exit code."""
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args(argv)
    return run_optimization(config=args.config)


if __name__ == "__main__":
    raise SystemExit(main())
