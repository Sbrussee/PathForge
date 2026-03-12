from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from pathbench.policy.base import PolicyBase
from pathbench.core.experiments.base import ComboConfig, Experiment
from pathbench.core.datasets.wsi_dataset import WSIDataset
from pathbench.core.io.h5.base import FileHandleH5
from pathbench.core.io.h5 import features as features_io
from pathbench.config.config import DatasetEntry
from pathbench.policy.feature_extraction import FeatureExtractionPolicy
from pathbench.benchmarking.registry import import_task_modules, get_task
from pathbench.utils.constants import DATASET_COL, SLIDE_ID_COL

logger = logging.getLogger(__name__)


class BenchmarkingPolicy(PolicyBase):
    """
    Benchmark orchestration policy.

    Flow:
    - resolve task from registry
    - build full benchmark combos
    - group combos by bag-defining parameters
    - per group:
        - check missing features from annotations
        - run feature extraction only for missing slides
        - build bag datasets once
        - run all inner combos for that bag setup
    """

    def __init__(self, experiment: Experiment):
        super().__init__(experiment)
        self.cfg = experiment.cfg

        task_name = self.cfg.experiment.task
        if task_name is None:
            raise ValueError("experiment.task must be set for benchmarking.")

        import_task_modules()
        self.task = build_task(task_name, self.experiment)

    def execute(self) -> dict[str, Any]:

        full_combos = self.experiment.build_combinations(self.task.get_grid_keys())
        if not full_combos:
            logger.warning("[Benchmark] No benchmark combinations found.")
            return {"status": "no_combos", "num_runs": 0}

        grouped_combos = self._group_combos_by_bag_source(full_combos)
        annotations_df = self.experiment.load_annotations()
        feature_policy = FeatureExtractionPolicy(self.experiment)

        logger.info(
            "[Benchmark] Task='%s' | %d full combos grouped into %d bag groups.",
            task_name,
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

            for ds_cfg in self.cfg.datasets:
                if ds_cfg.used_for == "ignore":
                    continue

                slide_ids_with_missing_features = self._find_slide_ids_with_missing_features(
                    ds_cfg=ds_cfg,
                    annotations_df=annotations_df,
                    combo_cfg=representative_combo,
                )

                if not slide_ids_with_missing_features:
                    logger.info(
                        "[Benchmark] Dataset '%s': all features already exist for current bag setup.",
                        ds_cfg.name,
                    )
                    continue

                logger.info(
                    "[Benchmark] Dataset '%s': extracting features for %d slides with missing features.",
                    ds_cfg.name,
                    len(slide_ids_with_missing_features),
                )

                try:
                    subset_dataset = self._build_subset_wsi_dataset_for_slide_ids(
                        ds_cfg=ds_cfg,
                        annotations_df=annotations_df,
                        slide_ids=slide_ids_with_missing_features,
                    )
                except FileNotFoundError as e:
                    raise RuntimeError(
                        f"Cannot continue benchmark for dataset '{ds_cfg.name}' "
                        f"(extractor={representative_combo.feature_extraction}, "
                        f"tile_px={representative_combo.tile_px}, "
                        f"tile_mpp={representative_combo.tile_mpp}) because some slides "
                        f"need feature extraction but are not available locally. {e}"
                    ) from e

                feature_policy.execute_dataset(
                    dataset=subset_dataset,
                    combo_cfg=representative_combo,
                )

            bag_datasets = self.experiment.build_bag_datasets(representative_combo)
            datasets_by_use = self._group_datasets_by_use(bag_datasets)

            for combo_index, combo_cfg in enumerate(group_combos, start=1):
                logger.info(
                    "[Benchmark] Running combo %d/%d in current bag group.",
                    combo_index,
                    len(group_combos),
                )

                self.task.execute(combo_cfg=combo_cfg, datasets_by_use=datasets_by_use)
                num_runs += 1

        logger.info("[Benchmark] Benchmarking complete. Total runs: %d", num_runs)

        return {
            "status": "benchmark_done",
            "num_runs": num_runs,
        }

    # ------------------------------------------------------------------
    # Combo grouping
    # ------------------------------------------------------------------

    def _group_combos_by_bag_source(
        self,
        combos: list[ComboConfig],
    ) -> dict[tuple[Any, Any, Any], list[ComboConfig]]:
        grouped: dict[tuple[Any, Any, Any], list[ComboConfig]] = {}

        for combo_cfg in combos:
            key = (
                combo_cfg.feature_extraction,
                combo_cfg.tile_px,
                combo_cfg.tile_mpp,
            )
            grouped.setdefault(key, []).append(combo_cfg)

        return grouped

    # ------------------------------------------------------------------
    # Missing-feature detection
    # ------------------------------------------------------------------

    def _find_slide_ids_with_missing_features(
        self,
        ds_cfg: DatasetEntry,
        annotations_df: pd.DataFrame,
        combo_cfg: ComboConfig,
    ) -> list[str]:
        dataset_df = annotations_df[annotations_df[DATASET_COL] == ds_cfg.name].copy()
        if dataset_df.empty:
            return []

        bag_id = self._build_bag_id(
            tile_px=int(combo_cfg.tile_px),
            tile_mpp=float(combo_cfg.tile_mpp),
        )
        extractor_name = str(combo_cfg.feature_extraction)
        artifacts_dir = Path(ds_cfg.artifacts_dir).expanduser().resolve()

        slide_ids = (
            dataset_df[SLIDE_ID_COL]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        slide_ids_with_missing_features: list[str] = []

        for slide_id in slide_ids:
            artifact_path = artifacts_dir / f"{slide_id}.h5"

            if not artifact_path.is_file():
                slide_ids_with_missing_features.append(slide_id)
                continue

            with FileHandleH5(artifact_path, mode="r") as slide_artifact:
                if not features_io.features_exist(
                    slide_artifact,
                    bag_id=bag_id,
                    extractor_name=extractor_name,
                ):
                    slide_ids_with_missing_features.append(slide_id)

        return slide_ids_with_missing_features

    def _build_subset_wsi_dataset_for_slide_ids(
        self,
        ds_cfg: DatasetEntry,
        annotations_df: pd.DataFrame,
        slide_ids: list[str],
    ) -> WSIDataset:
        slide_id_set = {str(slide_id) for slide_id in slide_ids}

        subset_annotations = annotations_df[
            (annotations_df[DATASET_COL] == ds_cfg.name)
            & (annotations_df[SLIDE_ID_COL].astype(str).isin(slide_id_set))
        ].copy()

        if subset_annotations.empty:
            raise FileNotFoundError(
                f"No annotation rows found for requested slide_ids in dataset '{ds_cfg.name}'."
            )

        subset_dataset = WSIDataset(ds_cfg, subset_annotations)

        found_slide_ids = {wsi.slide for wsi in subset_dataset.samples}
        missing_source_slide_ids = sorted(slide_id_set - found_slide_ids)

        if missing_source_slide_ids:
            raise FileNotFoundError(
                f"The following slides require feature extraction but are not available in "
                f"slides_dir='{ds_cfg.slides_dir}': {missing_source_slide_ids}"
            )

        return subset_dataset
    
    def _group_datasets_by_use(
        self,
        bag_datasets: list,
    ) -> dict[str, list]:
        """
        Group datasets by their ds_cfg.used_for value.

        This is generic and does not assume specific use types such as
        training/validation/testing.

        Args:
            bag_datasets: List of BagDataset objects.

        Returns:
            Dict mapping used_for -> list of datasets.
        """
        datasets_by_use: dict[str, list] = {}

        for dataset in bag_datasets:
            use = str(dataset.ds_cfg.used_for)
            datasets_by_use.setdefault(use, []).append(dataset)

        return datasets_by_use

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_bag_id(*, tile_px: int, tile_mpp: float) -> str:
        return f"{tile_px}px_{tile_mpp:g}mpp"