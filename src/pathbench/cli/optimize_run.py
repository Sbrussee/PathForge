from pathlib import Path

import typer

from ..config.config import Config
from ..core.experiments.base import OptimizationExperiment

def run_optimization(
    *,
    config: Path,
) -> int:
    cfg = Config.from_yaml(config)
    out = OptimizationExperiment(cfg).run()
    print(out)
    return 0


def run_command(
    config: Path = typer.Option(..., "--config", help="Path to YAML config."),
) -> None:
    raise SystemExit(run_optimization(config=config))


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args(argv)
    return run_optimization(config=args.config)


if __name__ == "__main__":
    raise SystemExit(main())
