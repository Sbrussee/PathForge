from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from pathbench.core.evaluation.orchestrator import EvaluationOrchestrator
from pathbench.core.evaluation.types import EvaluationRunContext
from pathbench.core.experiments.combinations import ComboConfig


class _FakeEvaluator:
    def __init__(self, run_contexts: list[EvaluationRunContext]) -> None:
        self._run_contexts = run_contexts

    def discover_runs(self) -> list[EvaluationRunContext]:
        return list(self._run_contexts)

    def load_run_data(self, run_context: EvaluationRunContext) -> object:
        return {"run_dir": str(run_context.run_dir)}


def test_evaluation_orchestrator_writes_run_level_metrics_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run_001"
    run_dir.mkdir(parents=True)
    combo_cfg = ComboConfig(tile_px=256, tile_px_params={})
    run_context = EvaluationRunContext(
        task_name="slide_retrieval",
        run_dir=run_dir,
        combo_cfg=combo_cfg,
        manifest={"num_queries": 1},
        label_column="category",
        aggregation_level="slide",
    )

    experiment = SimpleNamespace(
        cfg=SimpleNamespace(
            experiment=SimpleNamespace(task="slide_retrieval"),
            evaluation=SimpleNamespace(metrics=["hit_at_5"]),
        )
    )

    class _MetricSpec:
        canonical_name = "hit_at_k"

        @staticmethod
        def compute_fn(*args, **kwargs):
            _ = args, kwargs
            return {"k": 5, "macro": 1.0}

    monkeypatch.setattr(
        "pathbench.core.evaluation.orchestrator.import_task_evaluation_adapter_modules",
        lambda: None,
    )
    monkeypatch.setattr(
        "pathbench.core.evaluation.orchestrator.import_evaluation_metric_modules",
        lambda: None,
    )
    monkeypatch.setattr(
        "pathbench.core.evaluation.orchestrator.build_task_evaluation_adapter",
        lambda name, experiment: _FakeEvaluator([run_context]),
    )
    monkeypatch.setattr(
        "pathbench.core.evaluation.orchestrator.resolve_metric_request",
        lambda **kwargs: (
            _MetricSpec(),
            SimpleNamespace(raw_name="hit_at_5"),
        ),
    )

    output = EvaluationOrchestrator(experiment).evaluate()

    metrics_path = run_dir / "evaluation_metrics.json"
    assert output["status"] == "evaluation_done"
    assert output["num_runs"] == 1
    assert metrics_path.is_file()

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["task"] == "slide_retrieval"
    assert payload["metrics"] == {"hit_at_5": {"k": 5, "macro": 1.0}}
