from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from pathbench.benchmarking.registry import build_task, import_task_modules
from pathbench.config.config import DatasetEntry
from pathbench.core.datasets.factory import build_bag_datasets, build_wsi_dataset
from pathbench.core.datasets.utils import group_datasets_by_use
from pathbench.core.experiments.base import ComboConfig, Experiment
from pathbench.core.experiments.combo_ids import build_bag_id
from pathbench.core.features.utils import find_slides_with_missing_features
from pathbench.core.datasets.wsi_dataset import WSIDataset
from pathbench.policy.base import PolicyBase
from pathbench.policy.feature_extraction import FeatureExtractionPolicy

logger = logging.getLogger(__name__)


class BenchmarkingPolicy(PolicyBase):
    """
    Run the benchmarking workflow over all configured combinations.

    Inputs:
        experiment (Experiment):
            Experiment context with validated config, annotations, and project
            metadata.

    Outputs:
        dict[str, Any]:
            Benchmark execution summary with at least:
            - ``status`` (str)
            - ``num_runs`` (int)
    """

    BAG_SOURCE_KEYS = ("feature_extraction", "tile_px", "tile_mpp")

    def __init__(self, experiment: Experiment):
        super().__init__(experiment)

        self.task_name = self.cfg.experiment.task
        if self.task_name is None:
            raise ValueError("experiment.task must be set for benchmarking.")

        import_task_modules()
        self.task = build_task(self.task_name, self.experiment)
        self.feature_policy = FeatureExtractionPolicy(self.experiment)

    def execute(self, combo_cfg: ComboConfig | None = None) -> dict[str, Any]:
        """
        Execute the benchmarking workflow for one combo or the full task grid.

        Inputs:
            combo_cfg (ComboConfig | None):
                Optional single combination to execute. When omitted, the full
                task grid is executed.

        Outputs:
            dict[str, Any]:
                Summary dictionary with the number of executed task runs.

        """
        if combo_cfg is not None:
            return self.execute_combination(combo_cfg)

        full_combos = self.experiment.build_combinations(self.task.get_grid_keys())
        if not full_combos:
            logger.warning("[Benchmark] No benchmark combinations found.")
            return {"status": "no_combos", "num_runs": 0}

        grouped_combos = self._group_combos_by_bag_source(full_combos)
        annotations_df = self.experiment.load_annotations()

        logger.info(
            "[Benchmark] Task='%s' | %d full combos grouped into %d bag groups.",
            self.task_name,
            len(full_combos),
            len(grouped_combos),
        )

        num_runs = 0

        for group_index, group_combos in enumerate(grouped_combos.values(), start=1):
            representative_combo = group_combos[0]

            logger.info(
                "[Benchmark] === Bag group %d/%d | extractor=%s, tile_px=%s, tile_mpp=%s | %d inner combos ===",
                group_index,
                len(grouped_combos),
                representative_combo.feature_extraction,
                representative_combo.tile_px,
                representative_combo.tile_mpp,
                len(group_combos),
            )

            datasets_by_use = self.prepare_bag_group(
                combo_cfg=representative_combo,
                annotations_df=annotations_df,
            )   

            self._validate_dataset_uses(datasets_by_use=datasets_by_use)

            for combo_index, combo_cfg in enumerate(group_combos, start=1):
                self._execute_group_combination(
                    combo_cfg=combo_cfg,
                    datasets_by_use=datasets_by_use,
                    combo_index=combo_index,
                    num_group_combos=len(group_combos),
                )
                num_runs += 1

        logger.info("[Benchmark] Benchmarking complete. Total runs: %d", num_runs)
        return {"status": "benchmark_done", "num_runs": num_runs}

    def execute_combination(self, combo_cfg: ComboConfig) -> dict[str, Any]:
        """
        Execute exactly one full benchmark combination.

        Inputs:
            combo_cfg (ComboConfig):
                Full benchmark combination to execute.

        Outputs:
            dict[str, Any]:
                Summary dictionary for the single executed run.

        """
        bag_combo_cfg = self._project_bag_source_combo(combo_cfg)
        annotations_df = self.experiment.load_annotations()
        datasets_by_use = self.prepare_bag_group(
            combo_cfg=bag_combo_cfg,
            annotations_df=annotations_df,
        )
        task_output = self.task.execute(
            combo_cfg=combo_cfg,
            datasets_by_use=datasets_by_use,
        )
        return {
            "status": "benchmark_done",
            "num_runs": 1,
            "task_output": task_output,
        }

    def prepare_bag_group(
        self,
        *,
        combo_cfg: ComboConfig,
        annotations_df: pd.DataFrame | None = None,
    ) -> dict[str, list[Any]]:
        """
        Prepare reusable datasets for one bag-defining combination.

        Inputs:
            combo_cfg (ComboConfig):
                Combination containing at least:
                - ``feature_extraction`` (str)
                - ``tile_px`` (int)
                - ``tile_mpp`` (float)
            annotations_df (pandas.DataFrame | None):
                Optional annotations table to reuse in batch execution.

        Outputs:
            dict[str, list[Any]]:
                Bag datasets grouped by their configured use.

        """
        if annotations_df is None:
            annotations_df = self.experiment.load_annotations()

        self.complete_feature_extraction(
            combo_cfg=combo_cfg,
            annotations_df=annotations_df,
        )

        bag_datasets = build_bag_datasets(
            cfg=self.experiment.cfg,
            annotations_df=annotations_df,
            combo_cfg=combo_cfg,
        )
        return group_datasets_by_use(bag_datasets)

    def complete_feature_extraction(
        self,
        *,
        combo_cfg: ComboConfig,
        annotations_df: pd.DataFrame | None = None,
    ) -> None:
        """
        Ensure every non-ignored dataset has features for one bag-defining combo.

        Inputs:
            combo_cfg (ComboConfig):
                Combination containing at least:
                - ``feature_extraction`` (str)
                - ``tile_px`` (int)
                - ``tile_mpp`` (float)
            annotations_df (pandas.DataFrame | None):
                Optional annotations table to reuse across multiple calls.

        Outputs:
            None:
                Missing slide features are extracted in-place into dataset H5
                artifacts before bag datasets are constructed.

        """
        if annotations_df is None:
            annotations_df = self.experiment.load_annotations()

        allowed_uses = getattr(self.task, "allowed_dataset_uses", None)

        for ds_cfg in self.cfg.datasets:
            if ds_cfg.used_for == "ignore":
                continue
            if allowed_uses is not None and ds_cfg.used_for not in allowed_uses:
                continue

            missing_slide_ids = find_slides_with_missing_features(
                ds_cfg=ds_cfg,
                annotations_df=annotations_df,
                combo_cfg=combo_cfg,
            )

            if not missing_slide_ids:
                logger.info(
                    "[Benchmark] Dataset '%s': all features already exist for current bag setup.",
                    ds_cfg.name,
                )
                continue

            logger.info(
                "[Benchmark] Dataset '%s': extracting features for %d slides with missing features.",
                ds_cfg.name,
                len(missing_slide_ids),
            )

            subset_dataset = self._build_feature_extraction_dataset(
                ds_cfg=ds_cfg,
                annotations_df=annotations_df,
                missing_slide_ids=missing_slide_ids,
                combo_cfg=combo_cfg,
            )
            self.feature_policy.execute_dataset(dataset=subset_dataset, combo_cfg=combo_cfg)

    def _execute_group_combination(
        self,
        *,
        combo_cfg: ComboConfig,
        datasets_by_use: dict[str, list[Any]],
        combo_index: int,
        num_group_combos: int,
    ) -> dict[str, Any]:
        """
        Execute one benchmark combination inside an already prepared bag group.

        Inputs:
            combo_cfg (ComboConfig):
                Active benchmark combination for the task execution.
            datasets_by_use (dict[str, list[Any]]):
                Prepared bag datasets grouped by configured use.
            combo_index (int):
                One-based index of the current combo inside the bag group.
            num_group_combos (int):
                Total number of combos in the current bag group.

        Outputs:
            dict[str, Any]:
                Task-specific result dictionary returned by the benchmarking
                task implementation.

        """
        logger.info(
            "[Benchmark] Running combo %d/%d in current bag group.",
            combo_index,
            num_group_combos,
        )
        return self.task.execute(combo_cfg=combo_cfg, datasets_by_use=datasets_by_use)

    def _build_feature_extraction_dataset(
        self,
        *,
        ds_cfg: DatasetEntry,
        annotations_df: pd.DataFrame,
        missing_slide_ids: list[str],
        combo_cfg: ComboConfig,
    ) -> WSIDataset:
        """
        Build a restricted WSI dataset for missing-feature extraction.

        Inputs:
            ds_cfg (DatasetEntry):
                Dataset definition whose slides should be checked.
            annotations_df (pandas.DataFrame):
                Annotation table with one row per sample and a dataset identifier
                column.
            missing_slide_ids (list[str]):
                Unique slide identifiers requiring feature extraction. Shape:
                ``(n_missing,)``.
            combo_cfg (ComboConfig):
                Active bag-defining combo used only for clearer error reporting.

        Outputs:
            WSIDataset:
                Dataset restricted to slides whose features are missing.

        """
        try:
            return build_wsi_dataset(
                ds_cfg=ds_cfg,
                annotations_df=annotations_df,
                slide_ids=missing_slide_ids,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                f"Cannot continue benchmark for dataset '{ds_cfg.name}' "
                f"(extractor={combo_cfg.feature_extraction}, "
                f"tile_px={combo_cfg.tile_px}, "
                f"tile_mpp={combo_cfg.tile_mpp}) because some slides need "
                f"feature extraction but are not available locally. {error}"
            ) from error

    def _group_combos_by_bag_source(
        self,
        combos: list[ComboConfig],
    ) -> dict[str, list[ComboConfig]]:
        """
        Group combinations that share the same bag-defining source artifacts.

        Inputs:
            combos (list[ComboConfig]):
                Combination list. Shape: ``(n_combos,)``.

        Outputs:
            dict[str, list[ComboConfig]]:
                Mapping from bag identifier to the combos that can reuse the same
                extracted artifacts.

        """
        grouped: dict[str, list[ComboConfig]] = {}

        for combo_cfg in combos:
            bag_id = build_bag_id(combo_cfg)
            grouped.setdefault(bag_id, []).append(combo_cfg)

        return grouped

    def _project_bag_source_combo(self, combo_cfg: ComboConfig) -> ComboConfig:
        """
        Project a full combo onto the bag-defining fields.

        Inputs:
            combo_cfg (ComboConfig):
                Full benchmark combination.

        Outputs:
            ComboConfig:
                Reduced combination containing only the bag source fields.

        """
        bag_source_data = {
            key: getattr(combo_cfg, key)
            for key in self.BAG_SOURCE_KEYS
        }
        return ComboConfig(**bag_source_data)


    def _validate_dataset_uses(
        self,
        datasets_by_use: dict[str, list[Any]],
    ) -> None:
        allowed_uses = getattr(self.task, "allowed_dataset_uses", None)
        if allowed_uses is None:
            return

        invalid_uses = sorted(set(datasets_by_use) - set(allowed_uses))
        if invalid_uses:
            raise ValueError(
                f"Task '{self.task_name}' does not support dataset uses: {invalid_uses}. "
                f"Allowed uses: {sorted(allowed_uses)}"
            )