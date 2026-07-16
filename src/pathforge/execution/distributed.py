"""Plan and execute idempotent PathForge work units on local or HPC schedulers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import socket
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from pathforge.config.config import Config, StageResourceConfig
from pathforge.core.datasets.factory import build_wsi_datasets
from pathforge.core.experiments.base import Experiment
from pathforge.core.experiments.combinations import ComboConfig, build_combinations
from pathforge.core.tasks.registry import get_task, import_task_modules
from pathforge.policy.benchmarking import BenchmarkingPolicy

WorkStage = Literal["features", "benchmark"]


class WorkRecord(BaseModel):
    """One immutable, scheduler-neutral unit of distributed PathForge work."""

    schema_version: int = 1
    stage: WorkStage
    work_id: str
    config_path: str
    payload: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    """Metadata and manifest locations for one distributed experiment plan."""

    schema_version: int = 1
    plan_id: str
    created_at: str
    config_path: str
    plan_dir: str
    feature_manifest: str
    benchmark_manifest: str
    status_dir: str
    results_dir: str
    num_feature_jobs: int
    num_benchmark_jobs: int
    num_optimization_workers: int = 0


class WorkStatus(BaseModel):
    """Machine-readable lifecycle record written atomically by one worker."""

    schema_version: int = 1
    work_id: str
    stage: WorkStage
    state: Literal["running", "success", "failed", "skipped"]
    started_at: str
    finished_at: str | None = None
    hostname: str
    pid: int
    slurm_job_id: str | None = None
    slurm_array_task_id: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _canonical_hash(value: Any, *, length: int = 16) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def _write_json(path: Path, value: BaseModel | dict[str, Any]) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str))


def _write_jsonl(path: Path, records: list[WorkRecord]) -> None:
    lines = [record.model_dump_json() for record in records]
    _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def read_work_record(manifest_path: str | Path, index: int) -> WorkRecord:
    """Read one zero-based work record from a JSON-lines manifest."""

    if index < 0:
        raise ValueError("work index must be non-negative.")
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as handle:
        for current_index, line in enumerate(handle):
            if current_index == index:
                return WorkRecord.model_validate_json(line)
    raise IndexError(f"Manifest {path} has no work record at index {index}.")


def _config_snapshot(cfg: Config, destination: Path) -> None:
    serialized = yaml.safe_dump(
        cfg.model_dump(mode="json"),
        sort_keys=False,
        allow_unicode=True,
    )
    _atomic_write_text(destination, serialized)


def _feature_records(cfg: Config, config_path: Path) -> list[WorkRecord]:
    annotations = pd.read_csv(cfg.experiment.annotation_file)
    datasets = build_wsi_datasets(cfg=cfg, annotations_df=annotations)
    records: list[WorkRecord] = []
    for dataset in datasets:
        for wsi in dataset.samples:
            payload = {
                "dataset": dataset.name,
                "slide_id": wsi.slide,
                "input_path": str(wsi.path),
            }
            records.append(
                WorkRecord(
                    stage="features",
                    work_id=f"features__{_canonical_hash(payload)}",
                    config_path=str(config_path),
                    payload=payload,
                )
            )
    return sorted(records, key=lambda item: item.work_id)


def _benchmark_records(
    cfg: Config,
    config_path: Path,
    feature_records: list[WorkRecord],
) -> list[WorkRecord]:
    import_task_modules()
    if cfg.experiment.task is None:
        return []
    task_cls = get_task(cfg.experiment.task)
    combinations = build_combinations(cfg=cfg, keys=task_cls.get_grid_keys())
    feature_dependencies = [record.work_id for record in feature_records]
    records: list[WorkRecord] = []
    for combo in combinations:
        payload = {"combo": combo.to_dict()}
        records.append(
            WorkRecord(
                stage="benchmark",
                work_id=f"benchmark__{_canonical_hash(payload)}",
                config_path=str(config_path),
                payload=payload,
                dependencies=feature_dependencies,
            )
        )
    return sorted(records, key=lambda item: item.work_id)


def _slurm_directives(resource: StageResourceConfig, cfg: Config) -> list[str]:
    slurm = cfg.execution.slurm
    directives = [
        f"#SBATCH --cpus-per-task={resource.cpus}",
        f"#SBATCH --mem={resource.memory_gb}G",
        f"#SBATCH --time={resource.time}",
    ]
    if resource.gpus:
        directives.append(f"#SBATCH --gres=gpu:{resource.gpus}")
    for option, value in (
        ("partition", slurm.partition),
        ("account", slurm.account),
        ("qos", slurm.qos),
        ("constraint", slurm.constraint),
    ):
        if value:
            directives.append(f"#SBATCH --{option}={value}")
    directives.extend(f"#SBATCH {item}" for item in slurm.extra_directives)
    return directives


def _render_array_script(
    *,
    cfg: Config,
    plan_dir: Path,
    stage: WorkStage,
    manifest: Path,
    count: int,
) -> str:
    resource = (
        cfg.execution.resources.feature_extraction
        if stage == "features"
        else cfg.execution.resources.benchmarking
    )
    array_max = max(count - 1, 0)
    directives = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=pathforge-" + stage,
        f"#SBATCH --array=0-{array_max}%{cfg.execution.slurm.max_concurrent}",
        *_slurm_directives(resource, cfg),
        "set -euo pipefail",
        f'pathforge execution worker --plan "{plan_dir / "plan.json"}" '
        f'--stage {stage} --index "$SLURM_ARRAY_TASK_ID"',
    ]
    if count == 0:
        directives[-1] = f'echo "No {stage} work records in {manifest}"'
    return "\n".join(directives) + "\n"


def _render_slurm_files(
    cfg: Config,
    plan_dir: Path,
    feature_manifest: Path,
    benchmark_manifest: Path,
    feature_count: int,
    benchmark_count: int,
) -> None:
    slurm_dir = plan_dir / "slurm"
    feature_script = slurm_dir / "features.sbatch"
    benchmark_script = slurm_dir / "benchmark.sbatch"
    aggregate_script = slurm_dir / "aggregate.sbatch"
    _atomic_write_text(
        feature_script,
        _render_array_script(
            cfg=cfg,
            plan_dir=plan_dir,
            stage="features",
            manifest=feature_manifest,
            count=feature_count,
        ),
    )
    _atomic_write_text(
        benchmark_script,
        _render_array_script(
            cfg=cfg,
            plan_dir=plan_dir,
            stage="benchmark",
            manifest=benchmark_manifest,
            count=benchmark_count,
        ),
    )
    aggregate_lines = [
        "#!/usr/bin/env bash",
        "#SBATCH --job-name=pathforge-aggregate",
        *_slurm_directives(cfg.execution.resources.aggregation, cfg),
        "set -euo pipefail",
        f'pathforge execution aggregate --plan "{plan_dir / "plan.json"}"',
    ]
    _atomic_write_text(aggregate_script, "\n".join(aggregate_lines) + "\n")
    submit_lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    submit_lines.append(f'FEATURE_JOB=$(sbatch --parsable "{feature_script}")')
    if cfg.experiment.mode == "optimization":
        worker_count = math.ceil(
            cfg.optimization.trials / cfg.optimization.trials_per_worker
        )
        optimization_script = slurm_dir / "optimization.sbatch"
        optimization_lines = [
            "#!/usr/bin/env bash",
            "#SBATCH --job-name=pathforge-optimize",
            f"#SBATCH --array=0-{max(worker_count - 1, 0)}%{cfg.execution.slurm.max_concurrent}",
            *_slurm_directives(cfg.execution.resources.optimization, cfg),
            "set -euo pipefail",
            f"TOTAL_TRIALS={cfg.optimization.trials}",
            f"TRIALS_PER_WORKER={cfg.optimization.trials_per_worker}",
            "START=$((SLURM_ARRAY_TASK_ID * TRIALS_PER_WORKER))",
            "REMAINING=$((TOTAL_TRIALS - START))",
            "WORKER_TRIALS=$((REMAINING < TRIALS_PER_WORKER ? REMAINING : TRIALS_PER_WORKER))",
            f'pathforge optimize worker --config "{plan_dir / "config.snapshot.yaml"}" '
            '--trials "$WORKER_TRIALS"',
        ]
        _atomic_write_text(optimization_script, "\n".join(optimization_lines) + "\n")
        finalize_script = slurm_dir / "finalize-optimization.sbatch"
        finalize_lines = [
            "#!/usr/bin/env bash",
            "#SBATCH --job-name=pathforge-optuna-finalize",
            *_slurm_directives(cfg.execution.resources.aggregation, cfg),
            "set -euo pipefail",
            f'pathforge optimize finalize --config "{plan_dir / "config.snapshot.yaml"}"',
        ]
        _atomic_write_text(finalize_script, "\n".join(finalize_lines) + "\n")
        submit_lines.extend(
            [
                f'OPTIMIZATION_JOB=$(sbatch --parsable --dependency="afterok:${{FEATURE_JOB}}" "{optimization_script}")',
                f'sbatch --dependency="afterany:${{OPTIMIZATION_JOB}}" "{finalize_script}"',
            ]
        )
    else:
        submit_lines.extend(
            [
                f'BENCHMARK_JOB=$(sbatch --parsable --dependency="afterok:${{FEATURE_JOB}}" "{benchmark_script}")',
                f'sbatch --dependency="afterany:${{BENCHMARK_JOB}}" "{aggregate_script}"',
            ]
        )
    _atomic_write_text(slurm_dir / "submit.sh", "\n".join(submit_lines) + "\n")


def create_execution_plan(config_path: str | Path, plan_dir: str | Path) -> ExecutionPlan:
    """Materialize deduplicated feature and benchmark manifests plus SLURM files."""

    source_path = Path(config_path).expanduser().resolve()
    destination = Path(plan_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_path = destination / "config.snapshot.yaml"
    cfg = Config.from_yaml(source_path)
    if (
        cfg.experiment.mode == "optimization"
        and math.ceil(cfg.optimization.trials / cfg.optimization.trials_per_worker) > 1
        and cfg.optimization.storage is None
    ):
        raise ValueError(
            "Distributed optimization requires optimization.storage so workers "
            "can join the same Optuna study. Use PostgreSQL on a cluster."
        )
    features = _feature_records(cfg, snapshot_path)
    benchmarks = (
        _benchmark_records(cfg, snapshot_path, features)
        if cfg.experiment.mode == "benchmark"
        else []
    )
    feature_manifest = destination / "features.jsonl"
    benchmark_manifest = destination / "benchmark.jsonl"

    plan_payload = {
        "config": cfg.model_dump(mode="json"),
        "features": [record.model_dump(mode="json") for record in features],
        "benchmarks": [record.model_dump(mode="json") for record in benchmarks],
    }
    plan = ExecutionPlan(
        plan_id=f"plan__{_canonical_hash(plan_payload)}",
        created_at=_utc_now(),
        config_path=str(snapshot_path),
        plan_dir=str(destination),
        feature_manifest=str(feature_manifest),
        benchmark_manifest=str(benchmark_manifest),
        status_dir=str(destination / "status"),
        results_dir=str(destination / "results"),
        num_feature_jobs=len(features),
        num_benchmark_jobs=len(benchmarks),
        num_optimization_workers=(
            math.ceil(cfg.optimization.trials / cfg.optimization.trials_per_worker)
            if cfg.experiment.mode == "optimization"
            else 0
        ),
    )
    existing_plan_path = destination / "plan.json"
    if existing_plan_path.is_file():
        existing = load_plan(existing_plan_path)
        if existing.plan_id != plan.plan_id:
            raise ValueError(
                f"Plan directory {destination} already contains a different "
                f"execution plan ({existing.plan_id}). Choose another --output."
            )

    _config_snapshot(cfg, snapshot_path)
    _write_jsonl(feature_manifest, features)
    _write_jsonl(benchmark_manifest, benchmarks)
    _write_json(existing_plan_path, plan)
    _render_slurm_files(
        cfg,
        destination,
        feature_manifest,
        benchmark_manifest,
        len(features),
        len(benchmarks),
    )
    return plan


def load_plan(path: str | Path) -> ExecutionPlan:
    """Load and validate an execution plan JSON file."""

    return ExecutionPlan.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _status_path(plan: ExecutionPlan, work_id: str) -> Path:
    return Path(plan.status_dir) / f"{work_id}.json"


def _base_status(record: WorkRecord) -> WorkStatus:
    return WorkStatus(
        work_id=record.work_id,
        stage=record.stage,
        state="running",
        started_at=_utc_now(),
        hostname=socket.gethostname(),
        pid=os.getpid(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
    )


def _run_feature_record(record: WorkRecord) -> dict[str, Any]:
    from pathforge.cli.feature_extraction_slide import (
        run_feature_extraction_single_slide,
    )

    exit_code = run_feature_extraction_single_slide(
        config=Path(record.config_path),
        dataset=str(record.payload["dataset"]),
        input_path=Path(record.payload["input_path"]),
    )
    if exit_code != 0:
        raise RuntimeError(f"feature worker returned exit code {exit_code}.")
    return {
        "dataset": record.payload["dataset"],
        "slide_id": record.payload["slide_id"],
        "input_path": record.payload["input_path"],
    }


def _run_benchmark_record(record: WorkRecord, plan: ExecutionPlan) -> dict[str, Any]:
    incomplete: list[str] = []
    for dependency in record.dependencies:
        dependency_path = _status_path(plan, dependency)
        if not dependency_path.is_file():
            incomplete.append(dependency)
            continue
        dependency_status = WorkStatus.model_validate_json(
            dependency_path.read_text(encoding="utf-8")
        )
        if dependency_status.state != "success":
            incomplete.append(dependency)
    if incomplete:
        raise RuntimeError(
            "benchmark dependencies are not successful: " + ", ".join(incomplete)
        )

    cfg = Config.from_yaml(record.config_path)
    original_name = cfg.experiment.project_name
    cfg.experiment.project_name = f"{original_name}__{record.work_id}"
    base_root = Path(cfg.experiment.project_root or Path(plan.plan_dir) / "runs")
    cfg.experiment.project_root = str(base_root.resolve())
    experiment = Experiment(cfg)
    policy = BenchmarkingPolicy(experiment)
    combo = ComboConfig(**dict(record.payload["combo"]))
    output = policy.execute_combination(combo)
    return {
        "project_root": experiment.project_root,
        "combo": combo.to_dict(),
        "output": output,
    }


def execute_work_record(
    plan_path: str | Path,
    *,
    stage: WorkStage,
    index: int,
    resume: bool = True,
) -> WorkStatus:
    """Execute one manifest record and atomically persist its lifecycle status."""

    plan = load_plan(plan_path)
    manifest = plan.feature_manifest if stage == "features" else plan.benchmark_manifest
    record = read_work_record(manifest, index)
    status_path = _status_path(plan, record.work_id)
    if resume and status_path.is_file():
        existing = WorkStatus.model_validate_json(status_path.read_text(encoding="utf-8"))
        if existing.state == "success":
            return existing

    status = _base_status(record)
    _write_json(status_path, status)
    try:
        result = (
            _run_feature_record(record)
            if record.stage == "features"
            else _run_benchmark_record(record, plan)
        )
        status.state = "success"
        status.result = json.loads(json.dumps(result, default=str))
    except Exception as error:
        status.state = "failed"
        status.error = f"{type(error).__name__}: {error}"
        raise
    finally:
        status.finished_at = _utc_now()
        _write_json(status_path, status)
    return status


def _execute_index(arguments: tuple[str, WorkStage, int, bool]) -> dict[str, Any]:
    plan_path, stage, index, resume = arguments
    status = execute_work_record(plan_path, stage=stage, index=index, resume=resume)
    return status.model_dump(mode="json")


def execute_stage(
    plan_path: str | Path,
    *,
    stage: WorkStage,
    backend: Literal["local", "dask"] = "local",
    scheduler_address: str | None = None,
) -> list[dict[str, Any]]:
    """Execute every record in one stage through local processes or Dask."""

    plan = load_plan(plan_path)
    config = Config.from_yaml(plan.config_path)
    count = plan.num_feature_jobs if stage == "features" else plan.num_benchmark_jobs
    arguments = [
        (str(Path(plan_path).resolve()), stage, index, config.execution.resume)
        for index in range(count)
    ]
    if not arguments:
        return []
    if backend == "local":
        with ProcessPoolExecutor(max_workers=config.execution.max_workers) as executor:
            return list(executor.map(_execute_index, arguments))

    try:
        from dask.distributed import Client
    except ImportError as error:
        raise RuntimeError(
            "Dask execution requires dask.distributed. Install dask[distributed] "
            "or use execution.backend: local/slurm."
        ) from error

    client = Client(scheduler_address) if scheduler_address else Client()
    resource = (
        config.execution.resources.feature_extraction
        if stage == "features"
        else config.execution.resources.benchmarking
    )
    resources = {"GPU": 1} if resource.gpus else None
    try:
        futures = client.map(_execute_index, arguments, resources=resources)
        return list(client.gather(futures))
    finally:
        client.close()


def aggregate_plan(plan_path: str | Path) -> Path:
    """Aggregate isolated worker statuses into one deterministic CSV report."""

    plan = load_plan(plan_path)
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(plan.status_dir).glob("*.json")):
        status = WorkStatus.model_validate_json(path.read_text(encoding="utf-8"))
        row = {
            "work_id": status.work_id,
            "stage": status.stage,
            "state": status.state,
            "started_at": status.started_at,
            "finished_at": status.finished_at,
            "hostname": status.hostname,
            "slurm_job_id": status.slurm_job_id,
            "slurm_array_task_id": status.slurm_array_task_id,
            "error": status.error,
            "result": json.dumps(status.result, sort_keys=True, default=str),
        }
        rows.append(row)
    output_path = Path(plan.results_dir) / "distributed_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["work_id", "stage", "state"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
