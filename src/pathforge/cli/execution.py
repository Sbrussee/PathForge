"""CLI commands for distributed execution planning, workers, and aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from pathforge.execution.distributed import (
    WorkStage,
    aggregate_plan,
    create_execution_plan,
    execute_work_record,
    execute_stage,
    load_plan,
)


def plan_command(
    config: Path = typer.Option(..., "--config", help="Path to YAML config."),
    output: Path = typer.Option(..., "--output", help="Execution-plan directory."),
) -> None:
    """Materialize work manifests and ready-to-submit SLURM scripts."""

    plan = create_execution_plan(config, output)
    typer.echo(plan.model_dump_json(indent=2))


def worker_command(
    plan: Path = typer.Option(..., "--plan", help="Path to plan.json."),
    stage: WorkStage = typer.Option(..., "--stage", help="features or benchmark"),
    index: int = typer.Option(..., "--index", min=0, help="Zero-based manifest index."),
    no_resume: bool = typer.Option(False, "--no-resume", help="Rerun successful work."),
) -> None:
    """Execute exactly one idempotent manifest record."""

    status = execute_work_record(
        plan,
        stage=stage,
        index=index,
        resume=not no_resume,
    )
    typer.echo(status.model_dump_json(indent=2))


def aggregate_command(
    plan: Path = typer.Option(..., "--plan", help="Path to plan.json."),
) -> None:
    """Reduce isolated worker status files into a deterministic CSV report."""

    typer.echo(str(aggregate_plan(plan)))


def status_command(
    plan: Path = typer.Option(..., "--plan", help="Path to plan.json."),
) -> None:
    """Print counts of worker lifecycle states for one execution plan."""

    execution_plan = load_plan(plan)
    counts: dict[str, int] = {}
    for path in Path(execution_plan.status_dir).glob("*.json"):
        state = str(json.loads(path.read_text(encoding="utf-8")).get("state", "unknown"))
        counts[state] = counts.get(state, 0) + 1
    total = execution_plan.num_feature_jobs + execution_plan.num_benchmark_jobs
    counts["planned"] = max(total - sum(counts.values()), 0)
    typer.echo(json.dumps(counts, indent=2, sort_keys=True))


def run_command(
    plan: Path = typer.Option(..., "--plan", help="Path to plan.json."),
    stage: WorkStage = typer.Option(..., "--stage", help="features or benchmark"),
    backend: str = typer.Option("local", "--backend", help="local or dask"),
    scheduler_address: str | None = typer.Option(
        None,
        "--scheduler-address",
        help="Existing Dask scheduler address; omit to create a local client.",
    ),
) -> None:
    """Execute a complete manifest stage through local processes or Dask."""

    if backend not in {"local", "dask"}:
        raise typer.BadParameter("backend must be 'local' or 'dask'.")
    results = execute_stage(
        plan,
        stage=stage,
        backend=backend,
        scheduler_address=scheduler_address,
    )
    typer.echo(json.dumps(results, indent=2, sort_keys=True))
