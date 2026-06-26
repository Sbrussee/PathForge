from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pathbench.core.experiments.base import Experiment
from pathbench.core.visualization.registry import (
    build_task_visualization_adapter,
    import_task_visualization_adapter_modules,
)
from pathbench.core.visualization.types import VisualizationSummary
from pathbench.utils.constants import SLIDE_ID_COL


class VisualizationOrchestrator:
    """Run config-driven visualization for the configured task."""

    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment
        self.cfg = experiment.cfg

    def visualize(self) -> dict[str, Any]:
        """Render all configured visualizations for all discovered task runs."""

        task_name = self.cfg.experiment.task
        if task_name is None:
            raise ValueError("experiment.task must be set for visualization.")

        requested_visualizations = list(self.cfg.evaluation.visualization)
        if not requested_visualizations:
            return {
                "status": "no_visualizations",
                "num_runs": 0,
                "run_dirs": [],
                "created_files": [],
            }

        import_task_visualization_adapter_modules()
        visualization_adapter = build_task_visualization_adapter(
            task_name,
            self.experiment,
        )
        run_contexts = visualization_adapter.discover_runs()
        if not run_contexts:
            return {
                "status": "no_runs",
                "num_runs": 0,
                "run_dirs": [],
                "created_files": [],
            }

        subset_ids = self._load_subset_ids()
        created_files: list[str] = []
        processed_run_dirs: list[str] = []
        for run_context in run_contexts:
            run_files = visualization_adapter.render_run(
                run_context,
                requested_visualizations=requested_visualizations,
                subset_ids=subset_ids,
            )
            processed_run_dirs.append(str(run_context.run_dir))
            created_files.extend(str(path) for path in run_files)

        summary = VisualizationSummary(
            task_name=str(task_name),
            run_dirs=processed_run_dirs,
            created_files=created_files,
        )
        return {
            "status": "visualization_done",
            "num_runs": len(run_contexts),
            "run_dirs": list(summary.run_dirs),
            "created_files": list(summary.created_files),
        }

    def _load_subset_ids(self) -> set[str] | None:
        subset_path = self.cfg.evaluation.visualization_subset_file
        if subset_path is None:
            return None

        subset_file = Path(subset_path).expanduser().resolve()
        if not subset_file.is_file():
            raise FileNotFoundError(
                f"Visualization subset CSV not found: {subset_file}"
            )

        subset_df = pd.read_csv(subset_file)
        if SLIDE_ID_COL not in subset_df.columns:
            raise ValueError(
                "Visualization subset CSV must contain a 'slide' column."
            )

        subset_ids = {
            str(slide_id).strip()
            for slide_id in subset_df[SLIDE_ID_COL].tolist()
            if str(slide_id).strip()
        }
        if not subset_ids:
            raise ValueError(
                "Visualization subset CSV does not contain any non-empty slide IDs."
            )
        return subset_ids
