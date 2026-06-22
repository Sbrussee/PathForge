from __future__ import annotations

import copy
import logging
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from pathbench.core.tasks.registry import build_task
from pathbench.core.tasks.registry import import_task_modules
from pathbench.config.config import Config
from pathbench.config.config import DatasetEntry
from pathbench.core.datasets.factory import build_bag_datasets
from pathbench.core.datasets.factory import build_wsi_dataset
from pathbench.core.datasets.utils import group_datasets_by_use
from pathbench.core.datasets.wsi_dataset import WSIDataset
from pathbench.core.experiments.base import Experiment
from pathbench.core.experiments.combo_ids import build_bag_id
from pathbench.core.experiments.combinations import ComboConfig
from pathbench.core.experiments.combinations import build_combinations
from pathbench.core.features.utils import find_slides_with_missing_features
from pathbench.policy.base import PolicyBase
from pathbench.policy.feature_extraction import FeatureExtractionPolicy
from pathbench.policy.utils import apply_search_params
from pathbench.policy.utils import benchmark_search_space
from pathbench.policy.utils import build_bag_dataset_for_task
from pathbench.policy.utils import build_mil_model_for_config
from pathbench.policy.utils import collect_run_summary_row
from pathbench.policy.utils import experiment_output_root
from pathbench.policy.utils import infer_model_dimensions
from pathbench.policy.utils import metric_should_minimize
from pathbench.policy.utils import resolve_dataset_feature_dir
from pathbench.policy.utils import save_benchmark_visualizations
from pathbench.policy.utils import write_experiment_summary_csv
from pathbench.training.base import TrainerBase
from pathbench.utils.registries import LOSSES
from pathbench.utils.registries import TRAINERS

logger = logging.getLogger(__name__)


class BenchmarkingPolicy(PolicyBase):
    """Run PathBench benchmarking in experiment or legacy config mode."""

    def __init__(self, experiment: Experiment | Config | Any):
        """Initialize one benchmarking policy over an experiment-like context.

        Args:
            experiment: Either a fully prepared :class:`Experiment` or a legacy
                config-like object accepted by :class:`PolicyBase`.
        """

        super().__init__(experiment)
        self.task_name = self.cfg.experiment.task
        if self.task_name is None:
            raise ValueError("experiment.task must be set for benchmarking.")

        self._uses_experiment_context = hasattr(experiment, "load_annotations")
        self.task: Any | None = None
        self.feature_policy: FeatureExtractionPolicy | None = None
        self._summary_rows: list[dict[str, Any]] = []
        self._summary_output_path: Path | None = None
        self._summary_objective_metric = ""
        self._summary_minimize = True

        if self._uses_experiment_context:
            import_task_modules()
            self.task = build_task(self.task_name, self.experiment)
            self.feature_policy = FeatureExtractionPolicy(self.experiment)

    def execute(self) -> dict[str, Any]:
        """Run the configured benchmark workflow.

        Returns:
            dict[str, Any]: Summary containing at least ``status`` and
            ``num_runs``. In experiment mode, each run delegates to the
            task-specific benchmark executor. In legacy mode, the method
            evaluates expanded search-space configs directly.
        """

        if not self._uses_experiment_context:
            return self._execute_legacy_configs()

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
        for bag_group_index, (bag_id, combinations_for_bag_id) in enumerate(
            combinations_by_bag_id.items(),
            start=1,
        ):
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

            for combo_index, full_combo_cfg in enumerate(
                combinations_for_bag_id,
                start=1,
            ):
                logger.info(
                    "[Benchmark] Running combo %d/%d in current bag group.",
                    combo_index,
                    len(combinations_for_bag_id),
                )
                self.task.execute(
                    combo_cfg=full_combo_cfg,
                    datasets_by_use=datasets_by_use,
                )
                num_runs += 1

        logger.info("[Benchmark] Benchmarking complete. Total runs: %d", num_runs)
        return {"status": "benchmark_done", "num_runs": num_runs}

    def _execute_legacy_configs(self) -> dict[str, Any]:
        """Run the legacy config-grid benchmarking path.

        Returns:
            dict[str, Any]: Benchmark summary with one row per expanded legacy
            config combination.
        """

        configs = self._generate_configs()
        objective_metric = str(self.cfg.mil.best_epoch_based_on)
        minimize = metric_should_minimize(objective_metric)
        rows: list[dict[str, Any]] = []

        train_entry, val_entry = self._resolve_legacy_train_val_entries()
        for run_index, run_cfg in enumerate(configs):
            model_name = getattr(
                run_cfg,
                "_active_model_name",
                run_cfg.benchmark_parameters.mil[0],
            )
            loss_name = getattr(
                run_cfg,
                "_active_loss_name",
                run_cfg.benchmark_parameters.loss[0],
            )
            try:
                ds_train = build_bag_dataset_for_task(
                    run_cfg,
                    feature_dir=resolve_dataset_feature_dir(train_entry),
                    name="train",
                )
                ds_val = build_bag_dataset_for_task(
                    run_cfg,
                    feature_dir=resolve_dataset_feature_dir(val_entry),
                    name="val",
                )
                input_dim, output_dim = infer_model_dimensions(ds_train)
                model = build_mil_model_for_config(
                    run_cfg,
                    model_name=model_name,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    extra_kwargs={"dropout": run_cfg.mil.dropout_p},
                )
                trainer_class = TRAINERS.get("lightning")
                loss_factory = LOSSES.get(loss_name)
                trainer: TrainerBase = trainer_class(run_cfg)
                best_path, best_score = trainer.fit(
                    model,
                    ds_train,
                    ds_val,
                    loss_factory(),
                )
                rows.append(
                    collect_run_summary_row(
                        run_cfg,
                        run_index=run_index,
                        status="success",
                        objective_metric=objective_metric,
                        objective_value=float(best_score),
                        checkpoint_path=str(best_path),
                    )
                )
            except Exception as error:
                logger.exception("[Benchmark] Legacy benchmark run %d failed.", run_index)
                rows.append(
                    collect_run_summary_row(
                        run_cfg,
                        run_index=run_index,
                        status="failed",
                        objective_metric=objective_metric,
                        error=str(error),
                    )
                )

        self._summary_rows = rows
        self._summary_objective_metric = objective_metric
        self._summary_minimize = minimize
        self._summary_output_path = experiment_output_root(self.cfg) / "benchmark_results.csv"
        self._save_report()
        return {"status": "benchmark_done", "num_runs": len(configs)}

    def _resolve_legacy_train_val_entries(self) -> tuple[DatasetEntry, DatasetEntry]:
        """Resolve train/validation dataset entries for legacy benchmarking."""

        if not self.cfg.datasets:
            raise ValueError("benchmarking requires at least one configured dataset.")

        training_candidates = [
            dataset
            for dataset in self.cfg.datasets
            if str(dataset.used_for) in {"training", "all"}
        ]
        validation_candidates = [
            dataset
            for dataset in self.cfg.datasets
            if str(dataset.used_for) in {"validation", "testing", "all"}
        ]
        train_entry = training_candidates[0] if training_candidates else self.cfg.datasets[0]
        val_entry = validation_candidates[0] if validation_candidates else train_entry
        return train_entry, val_entry

    def _generate_configs(self) -> list[Config]:
        """Expand the configured benchmark search space into concrete configs."""

        search_space = benchmark_search_space(self.cfg)
        if not search_space:
            return [copy.deepcopy(self.cfg)]

        keys = list(search_space)
        values = [search_space[key] for key in keys]
        generated: list[Config] = []
        for combination in product(*values):
            config_copy = copy.deepcopy(self.cfg)
            params = dict(zip(keys, combination))
            apply_search_params(config_copy, params)
            generated.append(config_copy)
        return generated

    def _save_report(self) -> None:
        """Write summary CSV and benchmark visualizations when rows exist."""

        if self._summary_output_path is None:
            return
        write_experiment_summary_csv(
            self._summary_rows,
            output_path=self._summary_output_path,
            objective_metric=self._summary_objective_metric,
            minimize=self._summary_minimize,
        )
        save_benchmark_visualizations(
            self._summary_output_path,
            output_dir=self._summary_output_path.parent / "benchmark_visualizations",
            objective_metric=self._summary_objective_metric,
            minimize=self._summary_minimize,
            logger=logger,
        )

    def execute_combination(self, combo_cfg: ComboConfig) -> dict[str, Any]:
        """Run one fully specified combo through feature resolution and task execution.

        Args:
            combo_cfg: Benchmark combination describing one bag/model/task run.

        Returns:
            dict[str, Any]: One-run benchmark summary including the delegated
            task output.
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
        """Ensure all required slide-level bag features exist for one combo.

        Args:
            combo_cfg: Benchmark combination whose bag source must exist.
            annotations_df: Optional annotation table. When omitted, annotations
                are loaded from the experiment.
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
            if self.feature_policy is None:
                raise RuntimeError("FeatureExtractionPolicy is not available in legacy config mode.")
            self.feature_policy.execute_dataset(dataset=subset_dataset, combo_cfg=combo_cfg)

    def build_bag_datasets_for_combo(
        self,
        *,
        combo_cfg: ComboConfig,
        annotations_df: pd.DataFrame | None = None,
    ) -> list[Any]:
        """Build task-ready bag datasets for one benchmark combo."""

        if annotations_df is None:
            annotations_df = self.experiment.load_annotations()
        return build_bag_datasets(
            cfg=self.experiment.cfg,
            annotations_df=annotations_df,
            combo_cfg=combo_cfg,
            task=self.task_name,
        )

    def group_bag_datasets_by_use(
        self,
        bag_datasets: list[Any],
    ) -> dict[str, list[Any]]:
        """Group bag datasets by their configured ``used_for`` role."""

        return group_datasets_by_use(bag_datasets)

    def _build_feature_extraction_dataset(
        self,
        *,
        ds_cfg: DatasetEntry,
        annotations_df: pd.DataFrame,
        missing_slide_ids: list[str],
        combo_cfg: ComboConfig,
    ) -> WSIDataset:
        """Build a temporary WSI dataset restricted to slides missing features.

        Args:
            ds_cfg: Dataset configuration used to resolve source slides and
                artifact output paths.
            annotations_df: Full annotations table containing dataset rows.
            missing_slide_ids: Slide ids requiring feature extraction.
            combo_cfg: Active combo used only to enrich failure messaging.

        Returns:
            WSIDataset: Dataset containing only the missing slides.
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
        """Group full benchmark combos by canonical bag source identifier."""

        grouped: dict[str, list[ComboConfig]] = {}
        for combo_cfg in combos:
            bag_id = build_bag_id(combo_cfg)
            grouped.setdefault(bag_id, []).append(combo_cfg)
        return grouped

    def _validate_dataset_uses(
        self,
        datasets_by_use: dict[str, list[Any]],
    ) -> None:
        """Validate dataset-role keys against the active task contract."""

        allowed_uses = getattr(self.task, "allowed_dataset_uses", None)
        if allowed_uses is None:
            return
        invalid_uses = sorted(set(datasets_by_use) - set(allowed_uses))
        if invalid_uses:
            raise ValueError(
                f"Task '{self.task_name}' does not support dataset uses: {invalid_uses}. "
                f"Allowed uses: {sorted(allowed_uses)}"
            )
