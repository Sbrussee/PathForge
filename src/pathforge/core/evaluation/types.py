from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathforge.core.experiments.combinations import ComboConfig


@dataclass(frozen=True, slots=True)
class MetricRequest:
    """Parsed, immutable evaluation-metric request.

    Example:

    .. code-block:: python

        request = MetricRequest(
            raw_name="hit_at_5",
            canonical_name="hit_at_k",
            params={"k": 5},
        )

    """

    raw_name: str
    canonical_name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationRunContext:
    """Immutable evaluation context for one discovered run."""

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
    """JSON-serializable evaluation summary for one run."""

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
