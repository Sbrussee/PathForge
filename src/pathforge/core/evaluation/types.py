from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathforge.core.experiments.combinations import ComboConfig


@dataclass(frozen=True, slots=True)
class MetricRequest:
    """
    Parsed evaluation-metric request.

    Inputs:
    - `raw_name`: metric string requested in config.
    - `canonical_name`: registered metric or metric-family name.
    - `params`: parsed metric parameters.

    Returns:
    - Immutable request consumed by the evaluation orchestrator.

    Example:
    ```python
    request = MetricRequest(
        raw_name="hit_at_5",
        canonical_name="hit_at_k",
        params={"k": 5},
    )
    ```
    """

    raw_name: str
    canonical_name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationRunContext:
    """
    Evaluation context for one discovered run.

    Inputs:
    - `task_name`: registered task name.
    - `run_dir`: absolute path to the discovered run directory.
    - `combo_cfg`: benchmark combination used for run discovery.
    - `manifest`: parsed run manifest payload.
    - `label_column`: annotation label column used for truth resolution.
    - `aggregation_level`: aggregation level used by the task outputs.

    Returns:
    - Immutable run context consumed by task evaluators and metrics.
    """

    task_name: str
    run_dir: Path
    combo_cfg: ComboConfig
    manifest: dict[str, Any]
    label_column: str
    aggregation_level: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the run context."""
        return {
            "task_name": self.task_name,
            "run_dir": str(self.run_dir),
            "combo_cfg": self.combo_cfg.to_dict(),
            "manifest": dict(self.manifest),
            "label_column": self.label_column,
            "aggregation_level": self.aggregation_level,
        }


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """
    JSON-serializable evaluation summary for one run.

    Inputs:
    - `run_context`: evaluation run context carrying the shared run metadata.
    - `metrics`: mapping from requested metric string to metric payload.

    Returns:
    - Immutable summary that can be written directly to disk as JSON.
    """

    run_context: EvaluationRunContext
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary payload."""
        return {
            "task": self.run_context.task_name,
            "run_dir": str(self.run_context.run_dir),
            "label_column": self.run_context.label_column,
            "aggregation_level": self.run_context.aggregation_level,
            "combo_cfg": self.run_context.combo_cfg.to_dict(),
            "manifest": dict(self.run_context.manifest),
            "metrics": dict(self.metrics),
        }
