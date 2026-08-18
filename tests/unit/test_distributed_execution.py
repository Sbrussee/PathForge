"""Tests for scheduler-neutral plans, workers, aggregation, and SLURM rendering."""

from __future__ import annotations

import json
from pathlib import Path
import sys

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


def _write_feature_config(
    tmp_path: Path,
    *,
    slide_count: int = 1,
    slides_per_shard: int = 1,
    max_shards: int | None = None,
) -> Path:
    slides = tmp_path / "slides"
    slides.mkdir()
    slide_ids = [f"S{index}" for index in range(1, slide_count + 1)]
    for slide_id in slide_ids:
        (slides / f"{slide_id}.svs").touch()
    annotations = tmp_path / "annotations.csv"
    rows = [f"DS,{slide_id},P{index},case" for index, slide_id in enumerate(slide_ids)]
    annotations.write_text(
        "dataset,slide,patient,category\n" + "\n".join(rows) + "\n",
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
execution:
  slides_per_shard: {slides_per_shard}
{f"  max_shards: {max_shards}" if max_shards is not None else ""}
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
    assert record.payload["slides"][0]["slide_id"] == "S1"
    script = (tmp_path / "plan" / "slurm" / "features.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --array=0-0%20" in script
    pathforge_cli = Path(sys.prefix) / "bin" / "pathforge"
    assert f'"{pathforge_cli}" execution worker' in script


def test_execution_plan_groups_slides_into_sequential_feature_shards(
    tmp_path: Path,
) -> None:
    """Configured shard size reduces array jobs and preserves every slide."""

    config = _write_feature_config(tmp_path, slide_count=5, slides_per_shard=2)

    plan = create_execution_plan(config, tmp_path / "plan")

    assert plan.num_feature_jobs == 3
    shards = [
        read_work_record(plan.feature_manifest, index).payload["slides"]
        for index in range(plan.num_feature_jobs)
    ]
    assert [len(shard) for shard in shards] == [2, 2, 1]
    assert {slide["slide_id"] for shard in shards for slide in shard} == {
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
    }
    script = (tmp_path / "plan" / "slurm" / "features.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --array=0-2%20" in script


def test_execution_plan_rejects_non_positive_slides_per_shard(tmp_path: Path) -> None:
    """A shard must contain at least one slide."""

    config = _write_feature_config(tmp_path, slides_per_shard=1)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "slides_per_shard: 1", "slides_per_shard: 0"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="slides_per_shard"):
        create_execution_plan(config, tmp_path / "plan")


def test_execution_plan_caps_feature_shards_and_keeps_all_slides(
    tmp_path: Path,
) -> None:
    """``max_shards`` enlarges shards without losing or duplicating slides."""

    config = _write_feature_config(
        tmp_path,
        slide_count=10,
        slides_per_shard=2,
        max_shards=3,
    )

    plan = create_execution_plan(config, tmp_path / "plan")

    assert plan.num_feature_jobs == 3
    shards = [
        read_work_record(plan.feature_manifest, index).payload["slides"]
        for index in range(plan.num_feature_jobs)
    ]
    assert [len(shard) for shard in shards] == [4, 4, 2]
    slide_ids = [slide["slide_id"] for shard in shards for slide in shard]
    assert len(slide_ids) == 10
    assert len(set(slide_ids)) == 10
    script = (tmp_path / "plan" / "slurm" / "features.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --array=0-2%20" in script


def test_execution_plan_rejects_non_positive_max_shards(tmp_path: Path) -> None:
    """A configured shard cap must allow at least one scheduler work unit."""

    config = _write_feature_config(tmp_path, max_shards=1)
    config.write_text(
        config.read_text(encoding="utf-8").replace("max_shards: 1", "max_shards: 0"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_shards"):
        create_execution_plan(config, tmp_path / "plan")


def test_feature_shard_processes_slides_sequentially(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One feature worker invokes extraction for all slides in shard order."""

    config = _write_feature_config(tmp_path, slide_count=3, slides_per_shard=3)
    plan = create_execution_plan(config, tmp_path / "plan")
    calls: list[str] = []

    def fake_extract(*, config: Path, dataset: str, input_path: Path) -> int:
        calls.append(input_path.stem)
        return 0

    monkeypatch.setattr(
        "pathforge.cli.features_slide.run_feature_extraction_single_slide",
        fake_extract,
    )

    status = execute_work_record(
        tmp_path / "plan" / "plan.json", stage="features", index=0
    )

    expected = [
        slide["slide_id"]
        for slide in read_work_record(plan.feature_manifest, 0).payload["slides"]
    ]
    assert calls == expected
    assert [slide["slide_id"] for slide in status.result["slides"]] == expected


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
    pathforge_cli = Path(sys.prefix) / "bin" / "pathforge"
    assert f'"{pathforge_cli}" optimize worker' in worker_script
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
        return {"slide_id": str(record.payload["slides"][0]["slide_id"])}

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
