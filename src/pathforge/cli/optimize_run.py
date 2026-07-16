from pathlib import Path

import typer

from ..config.config import Config
from ..policy.optimization import OptimizationPolicy


def run_optimization(
    *,
    config: Path,
    trials: int | None = None,
    finalize: bool = True,
) -> int:
    """Run the optimization policy for one YAML config and return an exit code."""
    cfg = Config.from_yaml(config)
    OptimizationPolicy(cfg).execute(n_trials=trials, finalize=finalize)
    return 0


def run_command(
    config: Path = typer.Option(..., "--config", help="Path to YAML config."),
) -> None:
    """Typer command that runs optimization from the provided config option."""
    raise SystemExit(run_optimization(config=config))


def worker_command(
    config: Path = typer.Option(..., "--config", help="Path to YAML config."),
    trials: int | None = typer.Option(
        None,
        "--trials",
        min=1,
        help="Trials claimed by this worker; defaults to trials_per_worker.",
    ),
) -> None:
    """Join a shared Optuna study and execute this worker's trial allocation."""

    cfg = Config.from_yaml(config)
    worker_trials = trials or cfg.optimization.trials_per_worker
    OptimizationPolicy(cfg).execute(n_trials=worker_trials, finalize=False)


def finalize_command(
    config: Path = typer.Option(..., "--config", help="Path to YAML config."),
) -> None:
    """Load a shared Optuna study and write its final summaries and plots."""

    cfg = Config.from_yaml(config)
    OptimizationPolicy(cfg).execute(n_trials=0, finalize=True)


def main(argv: list[str] | None = None) -> int:
    """Argparse entry point for the optimization CLI; returns a process exit code."""
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args(argv)
    return run_optimization(config=args.config)


if __name__ == "__main__":
    raise SystemExit(main())
