from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pathforge.core.evaluation.registry import (
    build_task_evaluation_adapter,
    import_evaluation_metric_modules,
    import_task_evaluation_adapter_modules,
    resolve_metric_request,
)
from pathforge.core.evaluation.types import EvaluationSummary
from pathforge.core.experiments.base import Experiment


class EvaluationOrchestrator:
    """Run config-driven evaluation for the configured task."""

    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment
        self.cfg = experiment.cfg

    def evaluate(self) -> dict[str, Any]:
        """Evaluate all discovered runs for the configured task."""

        task_name = self.cfg.experiment.task
        if task_name is None:
            raise ValueError("experiment.task must be set for evaluation.")

        import_task_evaluation_adapter_modules()
        import_evaluation_metric_modules()

        evaluation_adapter = build_task_evaluation_adapter(
            task_name,
            self.experiment,
        )
        metric_specs_and_requests = [
            resolve_metric_request(task_name=task_name, raw_name=metric_name)
            for metric_name in self.cfg.evaluation.metrics
        ]

        run_contexts = evaluation_adapter.discover_runs()
        if not run_contexts:
            return {"status": "no_runs", "num_runs": 0, "run_dirs": []}

        evaluated_run_dirs: list[str] = []
        for run_context in run_contexts:
            normalized_data = evaluation_adapter.load_run_data(run_context)
            metric_payloads: dict[str, Any] = {}
            for metric_spec, metric_request in metric_specs_and_requests:
                metric_payloads[metric_request.raw_name] = metric_spec.compute_fn(
                    normalized_data,
                    request=metric_request,
                    run_context=run_context,
                )

            summary = EvaluationSummary(
                run_context=run_context,
                metrics=metric_payloads,
            )
            self._write_summary(summary)
            evaluated_run_dirs.append(str(run_context.run_dir))

        return {
            "status": "evaluation_done",
            "num_runs": len(run_contexts),
            "run_dirs": evaluated_run_dirs,
        }

    def _write_summary(self, summary: EvaluationSummary) -> None:
        """Write one run-level evaluation summary to disk."""

        run_dir = Path(summary.run_context.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "evaluation_metrics.json"
        metrics_path.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
