from __future__ import annotations

import json
from pathlib import Path

from pathbench.benchmarking.registry import get_task, import_task_modules
from pathbench.core.experiments.combo_ids import build_feature_name, build_tiling_id
from pathbench.core.experiments.combinations import build_combinations
from pathbench.core.visualization.base import TaskVisualizationAdapterBase
from pathbench.core.visualization.registry import task_visualization_adapter
from pathbench.core.visualization.types import VisualizationRunContext
from pathbench.slide_retrieval.io import (
    build_slide_retrieval_representation_root,
    build_slide_retrieval_output_root,
    resolve_slide_retrieval_results_path,
)
from pathbench.slide_retrieval.representation_strategies.registry import (
    build_representation_strategy,
)
from pathbench.slide_retrieval.representation_strategies.storage import (
    build_retrieval_representation_id,
)
from pathbench.slide_retrieval.visualization.service import (
    SlideRetrievalVisualizationService,
)


@task_visualization_adapter("slide_retrieval")
class SlideRetrievalVisualizationAdapter(TaskVisualizationAdapterBase):
    """
    Config-driven visualization adapter for slide retrieval runs.

    This adapter discovers completed slide-retrieval runs from benchmark
    combinations, then delegates actual rendering to the slide-retrieval
    visualization service.
    """

    task_name = "slide_retrieval"

    @classmethod
    def get_discovery_keys(cls) -> list[str]:
        import_task_modules()
        task_cls = get_task(cls.task_name)
        return task_cls.get_grid_keys()

    def discover_runs(self) -> list[VisualizationRunContext]:
        combos = build_combinations(
            cfg=self.cfg,
            keys=self.get_discovery_keys(),
        )

        run_contexts: list[VisualizationRunContext] = []
        for combo_cfg in combos:
            representation_root = build_slide_retrieval_representation_root(
                project_root=str(self.experiment.project_root),
                tiling_id=build_tiling_id(combo_cfg),
                feature_name=build_feature_name(combo_cfg),
                slide_representation=str(combo_cfg.get("retrieval_representation")),
            )
            search_root = build_slide_retrieval_output_root(
                project_root=str(self.experiment.project_root),
                tiling_id=build_tiling_id(combo_cfg),
                feature_name=build_feature_name(combo_cfg),
                slide_representation=str(combo_cfg.get("retrieval_representation")),
                search_method=str(combo_cfg.get("search_strategy")),
            )
            if search_root.exists():
                for run_dir in sorted(
                    path
                    for path in search_root.iterdir()
                    if path.is_dir() and path.name.startswith("run_")
                ):
                    manifest_path = run_dir / "manifest.json"
                    results_path = resolve_slide_retrieval_results_path(
                        run_dir / "query_results.xlsx"
                    )
                    if not manifest_path.is_file() or not results_path.is_file():
                        continue

                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    run_contexts.append(
                        VisualizationRunContext(
                            task_name=self.task_name,
                            run_dir=Path(run_dir).resolve(),
                            combo_cfg=combo_cfg,
                            manifest=manifest,
                            aggregation_level=str(manifest.get("aggregation_level", "")),
                        )
                    )

            representation_strategy = build_representation_strategy(
                str(combo_cfg.get("retrieval_representation")),
                params=combo_cfg.get_hyperparams("retrieval_representation"),
                bag_id=build_tiling_id(combo_cfg),
                config=getattr(self.experiment, "cfg", None),
            )
            representation_id = build_retrieval_representation_id(
                feature_extraction=build_feature_name(combo_cfg),
                retrieval_representation=str(combo_cfg.get("retrieval_representation")),
                params=representation_strategy.hyperparam_values(),
            )
            run_contexts.append(
                VisualizationRunContext(
                    task_name=self.task_name,
                    run_dir=representation_root.resolve(),
                    combo_cfg=combo_cfg,
                    manifest={
                        "tiling_id": build_tiling_id(combo_cfg),
                        "aggregation_level": str(self.cfg.experiment.aggregation_level),
                        "feature_extraction": build_feature_name(combo_cfg),
                        "slide_representation": str(combo_cfg.get("retrieval_representation")),
                        "slide_representation_params": dict(
                            representation_strategy.hyperparam_values()
                        ),
                        "search_method": str(combo_cfg.get("search_strategy")),
                        "representation_id": representation_id,
                        "synthetic_representation_only": True,
                        "representation_root": str(representation_root.resolve()),
                    },
                    aggregation_level=str(self.cfg.experiment.aggregation_level),
                )
            )

        return run_contexts

    def render_run(
        self,
        run_context: VisualizationRunContext,
        *,
        requested_visualizations: list[str],
        subset_ids: set[str] | None,
    ) -> list[Path]:
        aggregation_level = str(run_context.aggregation_level)
        if aggregation_level != "slide":
            raise ValueError(
                "Slide-retrieval visualization currently supports only "
                f"aggregation_level='slide'. Got {aggregation_level!r}."
            )

        service = SlideRetrievalVisualizationService(
            experiment=self.experiment,
            run_dir=run_context.run_dir,
            manifest=run_context.manifest,
            visualization_root=(
                run_context.run_dir
                if bool(run_context.manifest.get("synthetic_representation_only"))
                else None
            ),
        )
        if bool(run_context.manifest.get("synthetic_representation_only")):
            requested_visualizations = [
                name
                for name in requested_visualizations
                if str(name).strip() == "retrieval_representation"
            ]
        else:
            requested_visualizations = [
                name
                for name in requested_visualizations
                if str(name).strip() != "retrieval_representation"
            ]
        return service.render_requested_visualizations(
            requested_visualizations=requested_visualizations,
            subset_ids=subset_ids,
        )
