from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathforge.core.experiments.combinations import ComboConfig


@dataclass(frozen=True, slots=True)
class VisualizationRunContext:
    """
    Visualization context for one discovered run.

    Inputs:
    - `task_name`: registered task name.
    - `run_dir`: absolute path to the discovered run directory.
    - `combo_cfg`: benchmark combination used for run discovery.
    - `manifest`: parsed run manifest payload.
    - `aggregation_level`: aggregation level used by the task outputs.
    """

    task_name: str
    run_dir: Path
    combo_cfg: ComboConfig
    manifest: dict[str, Any]
    aggregation_level: str


@dataclass(frozen=True, slots=True)
class VisualizationSummary:
    """
    Summary of one config-driven visualization execution.

    Inputs:
    - `task_name`: task visualized.
    - `run_dirs`: run directories that were processed.
    - `created_files`: absolute PNG paths written to disk.
    """

    task_name: str
    run_dirs: list[str] = field(default_factory=list)
    created_files: list[str] = field(default_factory=list)

