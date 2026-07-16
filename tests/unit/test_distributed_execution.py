"""Tests for scheduler-neutral plans, workers, aggregation, and SLURM rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import pathforge.execution.distributed as distributed
from pathforge.execution.distributed import (
    ExecutionPlan,
    WorkRecord,
    aggregate_plan,
    create_execution_plan,
    execute_work_record,
    read_work_record,
)
from tests.conftest import DUMMY_FE


def _write_feature_config(tmp_path: Path) -> Path:
    slides = tmp_path / "slides"
    slides.mkdir()
    (slides / "S1.svs").touch()
    annotations = tmp_path / "annotations.csv"
    annotations.write_text(
        "dataset,slide,patient,category\nDS,S1,P1,case\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
experiment:
  project_name: distributed
  annotation_file: {annotations}
  project_root: {tmp_path / 'runs'}
  mode: feature_extraction
slide_processing:
  backend: lazyslide
datasets:
  - name: DS
    slides_dir: {slides}
    artifacts_dir: {tmp_path / 'artifacts'}
    used_for: training
benchmark_parameters:
  tile_px: [256]
  tile_mpp: [0.5]
  feature_extraction: [{DUMMY_FE}]
  mil: []
""".strip(),
        encoding="utf-8",
    )
    return config


def _write_optimization_config(tmp_path: Path, *, storage: str | None) -> Path:
    config = _write_feature_config(tmp_path)
    text = config.read_text(encoding="utf-8")
    text = text.replace(
        "  mode: feature_extraction",
        "  mode: optimization\n  task: classification",
    ).replace(
        "  mil: []",
        "  mil: [DummyMIL]\n  loss: [CrossEntropyLoss]",
    )
    storage_line = f"  storage: {storage}\n" if storage else ""
    text += (
        "\nmetrics:\n"
        "  classification_backend: native\n"
        "optimization:\n"
        "  study_name: distributed-study\n"
        "  objective_metric: val_loss\n"
        "  objective_mode: min\n"
        "  trials: 5\n"
        "  trials_per_worker: 2\n"
        f"{storage_line}"
    )
    config.write_text(text, encoding="utf-8")
    return config


def test_execution_plan_has_stable_ids_and_slurm_array(tmp_path: Path) -> None:
    config = _write_feature_config(tmp_path)

    first = create_execution_plan(config, tmp_path / "plan")
    second = create_execution_plan(config, tmp_path / "plan")

    assert first.plan_id == second.plan_id
    assert first.num_feature_jobs == 1
    record = read_work_record(first.feature_manifest, 0)
    assert record.stage == "features"
    assert record.payload["slide_id"] == "S1"
    script = (tmp_path / "plan" / "slurm" / "features.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --array=0-0%20" in script
    assert "pathforge execution worker" in script


def test_execution_plan_refuses_to_overwrite_different_plan(tmp_path: Path) -> None:
    """A populated plan directory cannot silently change experiment identity."""

    config = _write_feature_config(tmp_path)
    plan_dir = tmp_path / "plan"
    create_execution_plan(config, plan_dir)
    config.write_text(
        config.read_text(encoding="utf-8").replace("tile_px: [256]", "tile_px: [512]"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different execution plan"):
        create_execution_plan(config, plan_dir)


def test_optimization_plan_requires_shared_storage(tmp_path: Path) -> None:
    config = _write_optimization_config(tmp_path, storage=None)

    with pytest.raises(ValueError, match="optimization.storage"):
        create_execution_plan(config, tmp_path / "plan")


def test_optimization_plan_renders_parallel_workers_and_finalizer(tmp_path: Path) -> None:
    database = tmp_path / "study.db"
    config = _write_optimization_config(tmp_path, storage=f"sqlite:///{database}")

    plan = create_execution_plan(config, tmp_path / "plan")

    assert plan.num_optimization_workers == 3
    worker_script = (tmp_path / "plan" / "slurm" / "optimization.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --array=0-2%20" in worker_script
    assert "WORKER_TRIALS" in worker_script
    assert "pathforge optimize worker" in worker_script
    submit_script = (tmp_path / "plan" / "slurm" / "submit.sh").read_text(
        encoding="utf-8"
    )
    assert "afterok:${FEATURE_JOB}" in submit_script
    assert "afterany:${OPTIMIZATION_JOB}" in submit_script


def test_worker_status_is_atomic_and_success_is_resumable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _write_feature_config(tmp_path)
    plan = create_execution_plan(config, tmp_path / "plan")
    calls = 0

    def fake_run(record: WorkRecord) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"slide_id": str(record.payload["slide_id"])}

    monkeypatch.setattr(distributed, "_run_feature_record", fake_run)
    first = execute_work_record(tmp_path / "plan" / "plan.json", stage="features", index=0)
    second = execute_work_record(tmp_path / "plan" / "plan.json", stage="features", index=0)

    assert first.state == second.state == "success"
    assert calls == 1
    status_files = list(Path(plan.status_dir).glob("*.json"))
    assert len(status_files) == 1
    assert json.loads(status_files[0].read_text(encoding="utf-8"))["state"] == "success"


def test_failed_worker_persists_failure_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _write_feature_config(tmp_path)
    plan = create_execution_plan(config, tmp_path / "plan")
    monkeypatch.setattr(
        distributed,
        "_run_feature_record",
        lambda record: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        execute_work_record(tmp_path / "plan" / "plan.json", stage="features", index=0)

    status_path = next(Path(plan.status_dir).glob("*.json"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert "boom" in status["error"]


def test_aggregate_plan_collects_isolated_status_files(tmp_path: Path) -> None:
    plan_dir = tmp_path / "plan"
    status_dir = plan_dir / "status"
    status_dir.mkdir(parents=True)
    plan = ExecutionPlan(
        plan_id="plan__test",
        created_at="2026-01-01T00:00:00+00:00",
        config_path=str(plan_dir / "config.yaml"),
        plan_dir=str(plan_dir),
        feature_manifest=str(plan_dir / "features.jsonl"),
        benchmark_manifest=str(plan_dir / "benchmark.jsonl"),
        status_dir=str(status_dir),
        results_dir=str(plan_dir / "results"),
        num_feature_jobs=1,
        num_benchmark_jobs=0,
    )
    (plan_dir / "plan.json").write_text(plan.model_dump_json(), encoding="utf-8")
    (status_dir / "features__x.json").write_text(
        json.dumps(
            {
                "work_id": "features__x",
                "stage": "features",
                "state": "success",
                "started_at": "2026-01-01T00:00:00+00:00",
                "finished_at": "2026-01-01T00:01:00+00:00",
                "hostname": "node",
                "pid": 1,
                "result": {"slide_id": "S1"},
            }
        ),
        encoding="utf-8",
    )

    output = aggregate_plan(plan_dir / "plan.json")

    assert output.is_file()
    assert "features__x" in output.read_text(encoding="utf-8")
