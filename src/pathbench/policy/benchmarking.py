from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from pathbench.benchmarking.registry import build_task, import_task_modules
from pathbench.config.config import DatasetEntry
from pathbench.core.datasets.factory import build_bag_datasets, build_wsi_dataset
from pathbench.core.datasets.utils import group_datasets_by_use
from pathbench.core.experiments.base import Experiment
from pathbench.core.experiments.combinations import ComboConfig, build_combinations
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

    def __init__(self, experiment: Experiment):
        super().__init__(experiment)

        self.task_name = self.cfg.experiment.task
        if self.task_name is None:
            raise ValueError("experiment.task must be set for benchmarking.")

        import_task_modules()
        self.task = build_task(self.task_name, self.experiment)
        self.feature_policy = FeatureExtractionPolicy(self.experiment)

    def execute(self) -> dict[str, Any]:
        """
        Execute the full benchmarking workflow over the task grid.

        Outputs:
            dict[str, Any]:
                Summary dictionary with the number of executed task runs.

        """
        combinations = build_combinations(
            cfg=self.experiment.cfg,
            keys=self.task.get_grid_keys(),
        )
        if not combinations:
            logger.warning("[Benchmark] No benchmark combinations found.")
            return {"status": "no_combos", "num_runs": 0}

        combinations_by_bag_id = self._group_combos_by_bag_source(combinations)
        annotations_df = self.experiment.load_annotations()

        logger.info(
            "[Benchmark] Task='%s' | %d full combos grouped into %d bag groups.",
            self.task_name,
            len(combinations),
            len(combinations_by_bag_id),
        )

        num_runs = 0

        # Reuse one prepared bag dataset set per bag_id, then run all full combos in that group.
        for bag_group_index, (bag_id, combinations_for_bag_id) in enumerate(combinations_by_bag_id.items(), start=1):
            bag_source_combo = combinations_for_bag_id[0]

            logger.info(
                "[Benchmark] === Bag group %d/%d | bag_id=%s | extractor=%s, tile_px=%s, tile_mpp=%s | %d full combos ===",
                bag_group_index,
                len(combinations_by_bag_id),
                bag_id,
                bag_source_combo.feature_extraction,
                bag_source_combo.tile_px,
                bag_source_combo.tile_mpp,
                len(combinations_for_bag_id),
            )

            self.ensure_bag_features_exist(
                combo_cfg=bag_source_combo,
                annotations_df=annotations_df,
            )
            bag_datasets = self.build_bag_datasets_for_combo(
                combo_cfg=bag_source_combo,
                annotations_df=annotations_df,
            )
            datasets_by_use = self.group_bag_datasets_by_use(bag_datasets)

            self._validate_dataset_uses(datasets_by_use=datasets_by_use)

            for combo_index, full_combo_cfg in enumerate(combinations_for_bag_id, start=1,):
                logger.info(
                    "[Benchmark] Running combo %d/%d in current bag group.",
                    combo_index,
                    len(combinations_for_bag_id),
                )
                self.task.execute(combo_cfg=full_combo_cfg, datasets_by_use=datasets_by_use)
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
        annotations_df = self.experiment.load_annotations()
        self.ensure_bag_features_exist(
            combo_cfg=combo_cfg,
            annotations_df=annotations_df,
        )
        bag_datasets = self.build_bag_datasets_for_combo(
            combo_cfg=combo_cfg,
            annotations_df=annotations_df,
        )
        datasets_by_use = self.group_bag_datasets_by_use(bag_datasets)
        task_output = self.task.execute(
            combo_cfg=combo_cfg,
            datasets_by_use=datasets_by_use,
        )
        return {
            "status": "benchmark_done",
            "num_runs": 1,
            "task_output": task_output,
        }

    def ensure_bag_features_exist(
        self,
        *,
        combo_cfg: ComboConfig,
        annotations_df: pd.DataFrame | None = None,
    ) -> None:
        """
        Ensure bag-source features exist for one bag-defining combination.

        Inputs:
            combo_cfg (ComboConfig):
                Combination containing at least:
                - ``feature_extraction`` (str)
                - ``tile_px`` (int)
                - ``tile_mpp`` (float)
            annotations_df (pandas.DataFrame | None):
                Optional annotations table to reuse in batch execution.

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

    def build_bag_datasets_for_combo(
        self,
        *,
        combo_cfg: ComboConfig,
        annotations_df: pd.DataFrame | None = None,
    ) -> list[Any]:
        """
        Build bag datasets for one bag-defining combination.

        Inputs:
            combo_cfg (ComboConfig):
                Combination containing at least:
                - ``feature_extraction`` (str)
                - ``tile_px`` (int)
                - ``tile_mpp`` (float)
            annotations_df (pandas.DataFrame | None):
                Optional annotations table to reuse in batch execution.

        Outputs:
            list[Any]:
                Bag datasets for the provided bag-source combination.

        """
        if annotations_df is None:
            annotations_df = self.experiment.load_annotations()

        return build_bag_datasets(
            cfg=self.experiment.cfg,
            annotations_df=annotations_df,
            combo_cfg=combo_cfg,
        )

    def group_bag_datasets_by_use(
        self,
        bag_datasets: list[Any],
    ) -> dict[str, list[Any]]:
        """
        Group bag datasets by their configured use labels.

        Inputs:
            bag_datasets (list[Any]):
                Bag datasets built for one bag-defining combination.

        Outputs:
            dict[str, list[Any]]:
                Mapping from dataset use labels to bag dataset lists.

        """
        return group_datasets_by_use(bag_datasets)
        

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
