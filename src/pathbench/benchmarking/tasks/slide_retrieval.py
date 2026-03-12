from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pathbench.benchmarking.registry import register_task
from pathbench.benchmarking.tasks.base import TaskBase
from pathbench.core.experiments.base import ComboConfig

# Assumed helpers/registries to add or connect
from pathbench.slide_retrieval.representation.registry import build_slide_representation_method
from pathbench.slide_retrieval.search.registry import build_search_method
from pathbench.slide_retrieval.evaluation import evaluate_retrieval_metrics


logger = logging.getLogger(__name__)


@register_task("slide_retrieval")
class SlideRetrievalTask(TaskBase):
    """
    Slide-level retrieval task.

    Flow:
    - choose atlas/query datasets from datasets_by_use
    - build one representation per sample from BagDataset
    - run search
    - evaluate retrieval results
    - save outputs
    """

    # Rename these to match your config field names
    grid_keys = [
        "feature_extraction",
        "tile_px",
        "tile_mpp",
        "slide_representation_method",
        "search_method",
    ]

    def run(
        self,
        combo_cfg: ComboConfig,
        datasets_by_use: dict[str, list[Any]],
    ) -> None:
        atlas_datasets, query_datasets = self._resolve_atlas_and_query_datasets(datasets_by_use)

        representation_method = self._build_representation_method(combo_cfg)
        atlas_items = self._build_sample_representations(
            datasets=atlas_datasets,
            representation_method=representation_method,
            combo_cfg=combo_cfg,
        )
        query_items = self._build_sample_representations(
            datasets=query_datasets,
            representation_method=representation_method,
            combo_cfg=combo_cfg,
        )

        search_method = self._build_search_method(combo_cfg)
        results = search_method.search(
            query_items=query_items,
            atlas_items=atlas_items,
        )

        metric_names = list(getattr(self.cfg.experiment, "evaluation", []) or [])
        metrics = evaluate_retrieval_metrics(results, metric_names) if metric_names else {}

        self._save_outputs(
            combo_cfg=combo_cfg,
            results=results,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Dataset selection
    # ------------------------------------------------------------------

    def _resolve_atlas_and_query_datasets(
        self,
        datasets_by_use: dict[str, list[Any]],
    ) -> tuple[list[Any], list[Any]]:
        if "training" in datasets_by_use:
            atlas_datasets = datasets_by_use["training"]
        elif "all" in datasets_by_use:
            atlas_datasets = datasets_by_use["all"]
        else:
            atlas_datasets = [ds for group in datasets_by_use.values() for ds in group]

        if "testing" in datasets_by_use:
            query_datasets = datasets_by_use["testing"]
        elif "validation" in datasets_by_use:
            query_datasets = datasets_by_use["validation"]
        elif "all" in datasets_by_use:
            query_datasets = datasets_by_use["all"]
        else:
            query_datasets = [ds for group in datasets_by_use.values() for ds in group]

        if not atlas_datasets:
            raise ValueError("No atlas datasets available for slide retrieval.")
        if not query_datasets:
            raise ValueError("No query datasets available for slide retrieval.")

        return atlas_datasets, query_datasets

    # ------------------------------------------------------------------
    # Representation building
    # ------------------------------------------------------------------

    def _build_representation_method(self, combo_cfg: ComboConfig) -> Any:
        method_name = str(combo_cfg.slide_representation_method)
        return build_slide_representation_method(
            name=method_name,
            config=self.cfg,
            combo_cfg=combo_cfg,
        )

    def _build_sample_representations(
        self,
        datasets: list[Any],
        representation_method: Any,
        combo_cfg: ComboConfig,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []

        for dataset in datasets:
            for index in range(len(dataset)):
                bag, category = dataset[index]
                sample = dataset.get_sample(index)

                representation = representation_method.run(
                    bag=bag,
                    sample=sample,
                    combo_cfg=combo_cfg,
                )

                items.append(
                    {
                        "sample_id": sample.sample_id,
                        "patient_id": sample.patient_id,
                        "case_id": sample.case_id,
                        "category": category,
                        "slide_ids": sample.slide_ids,
                        "dataset_name": dataset.name,
                        "dataset_use": dataset.ds_cfg.used_for,
                        "metadata": sample.metadata,
                        "representation": representation,
                    }
                )

        return items

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _build_search_method(self, combo_cfg: ComboConfig) -> Any:
        method_name = str(combo_cfg.search_method)
        return build_search_method(
            name=method_name,
            config=self.cfg,
            combo_cfg=combo_cfg,
        )

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _save_outputs(
        self,
        combo_cfg: ComboConfig,
        results: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> None:
        output_dir = self._build_output_dir(combo_cfg)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "retrieval_results.json"
        metrics_path = output_dir / "retrieval_metrics.json"
        combo_path = output_dir / "combo_config.json"

        with results_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=self._json_default)

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=self._json_default)

        with combo_path.open("w", encoding="utf-8") as f:
            json.dump(combo_cfg.to_dict(), f, indent=2, default=self._json_default)

        logger.info("[SlideRetrievalTask] Saved outputs to %s", output_dir)

    def _build_output_dir(self, combo_cfg: ComboConfig) -> Path:
        if self.experiment.project_root is None:
            raise RuntimeError("project_root is not set.")

        combo_id = self._build_combo_id(combo_cfg)
        return Path(self.experiment.project_root) / "slide_retrieval" / combo_id

    @staticmethod
    def _build_combo_id(combo_cfg: ComboConfig) -> str:
        parts: list[str] = []

        for key, value in combo_cfg.to_dict().items():
            parts.append(f"{key}-{value}")

        return "__".join(parts)

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)