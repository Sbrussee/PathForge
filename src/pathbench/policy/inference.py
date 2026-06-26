from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from pathbench.core.tasks.registry import build_task, import_task_modules
from pathbench.config.config import DatasetEntry
from pathbench.core.datasets.factory import build_bag_dataset, build_wsi_dataset
from pathbench.core.datasets.utils import group_datasets_by_use
from pathbench.core.datasets.wsi_dataset import WSIDataset
from pathbench.core.experiments.base import Experiment
from pathbench.core.experiments.combinations import ComboConfig, build_combinations
from pathbench.core.experiments.combo_ids import build_bag_id
from pathbench.core.features.utils import find_slides_with_missing_features
from pathbench.policy.base import PolicyBase
from pathbench.policy.feature_extraction import FeatureExtractionPolicy
from pathbench.utils.constants import (
    CASE_ID_COL,
    CATEGORY_COL,
    DATASET_COL,
    PATIENT_ID_COL,
    SLIDE_ID_COL,
)

logger = logging.getLogger(__name__)


class InferencePolicy(PolicyBase):
    """
    Run task-specific inference over an explicit CSV-selected input batch.

    Inputs:
        experiment (Experiment):
            Experiment context with validated config, project metadata, and
            annotations.

    Outputs:
        dict[str, Any]:
            Inference execution summary with the timestamped invocation folder
            and per-combination task outputs.
    """

    def __init__(self, experiment: Experiment):
        super().__init__(experiment)

        self.task_name = self.cfg.experiment.task
        if self.task_name is None:
            raise ValueError("experiment.task must be set for inference.")

        import_task_modules()
        self.task = build_task(self.task_name, self.experiment)
        self.feature_policy = FeatureExtractionPolicy(self.experiment)

    def execute(self, *, input_csv: Path) -> dict[str, Any]:
        """
        Execute inference for every configured inference combination.

        Inputs:
            input_csv (Path):
                CSV file selecting the slides to use as inference/query inputs.
                Required columns are ``dataset`` and ``slide``.

        Outputs:
            dict[str, Any]:
                Summary with ``status``, ``output_dir``, ``num_runs``, and
                ``task_outputs``.
        """
        input_df = self._load_and_validate_input_csv(input_csv)
        annotations_df = self.experiment.load_annotations()
        input_annotations_df = self._build_input_annotations(
            input_df=input_df,
            annotations_df=annotations_df,
        )
        inference_run_root = self._create_inference_run_root()
        self._write_invocation_inputs(
            inference_run_root=inference_run_root,
            input_csv=input_csv,
        )

        combinations = build_combinations(
            cfg=self.experiment.cfg,
            keys=self.task.get_inference_grid_keys(),
        )
        if not combinations:
            logger.warning("[Inference] No inference combinations found.")
            return {
                "status": "no_combos",
                "output_dir": str(inference_run_root),
                "num_runs": 0,
                "task_outputs": [],
            }

        logger.info(
            "[Inference] Task='%s' | %d combo(s) | output_dir=%s",
            self.task_name,
            len(combinations),
            inference_run_root,
        )

        task_outputs: list[dict[str, Any]] = []
        for combo_index, combo_cfg in enumerate(combinations, start=1):
            logger.info(
                "[Inference] Running combo %d/%d | bag_id=%s",
                combo_index,
                len(combinations),
                build_bag_id(combo_cfg),
            )
            datasets_by_use = self._prepare_datasets_for_combo(
                combo_cfg=combo_cfg,
                annotations_df=annotations_df,
                input_annotations_df=input_annotations_df,
            )
            task_output = self.task.inference(
                combo_cfg=combo_cfg,
                datasets_by_use=datasets_by_use,
                inference_run_root=inference_run_root,
            )
            task_outputs.append(task_output)

        self._write_invocation_manifest(
            inference_run_root=inference_run_root,
            input_csv=input_csv,
            combinations=combinations,
            task_outputs=task_outputs,
        )
        return {
            "status": "inference_done",
            "output_dir": str(inference_run_root),
            "num_runs": len(task_outputs),
            "task_outputs": task_outputs,
        }

    def _load_and_validate_input_csv(self, input_csv: Path) -> pd.DataFrame:
        input_path = Path(input_csv)
        if not input_path.is_file():
            raise FileNotFoundError(f"Inference input CSV not found: {input_path}")

        input_df = pd.read_csv(input_path)
        required_columns = {DATASET_COL, SLIDE_ID_COL}
        missing_columns = sorted(required_columns - set(input_df.columns))
        if missing_columns:
            raise ValueError(
                f"Inference input CSV is missing required columns: {missing_columns}"
            )
        if input_df.empty:
            raise ValueError("Inference input CSV must contain at least one row.")

        input_df = input_df.copy()
        input_df[DATASET_COL] = input_df[DATASET_COL].astype(str)
        input_df[SLIDE_ID_COL] = input_df[SLIDE_ID_COL].astype(str)

        configured_dataset_names = {ds_cfg.name for ds_cfg in self.cfg.datasets}
        unknown_dataset_names = sorted(
            set(input_df[DATASET_COL].dropna().astype(str)) - configured_dataset_names
        )
        if unknown_dataset_names:
            raise ValueError(
                "Inference input CSV references unknown dataset(s): "
                f"{unknown_dataset_names}. Configured datasets: "
                f"{sorted(configured_dataset_names)}"
            )
        return input_df

    def _build_input_annotations(
        self,
        *,
        input_df: pd.DataFrame,
        annotations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        selected_annotation_rows = self._select_existing_annotation_rows(
            input_df=input_df,
            annotations_df=annotations_df,
        )
        missing_input_rows = self._select_rows_missing_from_annotations(
            input_df=input_df,
            existing_rows=selected_annotation_rows,
        )
        synthesized_rows = self._synthesize_missing_annotation_rows(missing_input_rows)

        if synthesized_rows.empty:
            return selected_annotation_rows.reset_index(drop=True)

        all_columns = sorted(set(selected_annotation_rows.columns) | set(synthesized_rows.columns))
        return pd.concat(
            [
                selected_annotation_rows.reindex(columns=all_columns),
                synthesized_rows.reindex(columns=all_columns),
            ],
            ignore_index=True,
        )

    def _select_existing_annotation_rows(
        self,
        *,
        input_df: pd.DataFrame,
        annotations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if annotations_df.empty:
            return pd.DataFrame(columns=annotations_df.columns)

        key_df = input_df[[DATASET_COL, SLIDE_ID_COL]].drop_duplicates()
        merged = annotations_df.merge(
            key_df,
            on=[DATASET_COL, SLIDE_ID_COL],
            how="inner",
        )
        return merged.copy()

    def _select_rows_missing_from_annotations(
        self,
        *,
        input_df: pd.DataFrame,
        existing_rows: pd.DataFrame,
    ) -> pd.DataFrame:
        existing_keys = {
            (str(row[DATASET_COL]), str(row[SLIDE_ID_COL]))
            for _, row in existing_rows[[DATASET_COL, SLIDE_ID_COL]].drop_duplicates().iterrows()
        }
        missing_mask = [
            (str(row[DATASET_COL]), str(row[SLIDE_ID_COL])) not in existing_keys
            for _, row in input_df.iterrows()
        ]
        return input_df.loc[missing_mask].copy()

    def _synthesize_missing_annotation_rows(
        self,
        missing_input_rows: pd.DataFrame,
    ) -> pd.DataFrame:
        if missing_input_rows.empty:
            return pd.DataFrame()

        aggregation_level = str(self.cfg.experiment.aggregation_level)
        synthesized: list[dict[str, Any]] = []
        for _, row in missing_input_rows.iterrows():
            slide_id = str(row[SLIDE_ID_COL])
            output_row = row.to_dict()
            output_row[DATASET_COL] = str(row[DATASET_COL])
            output_row[SLIDE_ID_COL] = slide_id
            output_row.setdefault(CATEGORY_COL, "unknown")

            if PATIENT_ID_COL not in output_row or pd.isna(output_row[PATIENT_ID_COL]):
                if aggregation_level == "patient":
                    raise ValueError(
                        "Inference input CSV must include a non-empty "
                        f"'{PATIENT_ID_COL}' column for slides missing from "
                        "project annotations when aggregation_level='patient'."
                    )
                output_row[PATIENT_ID_COL] = slide_id

            if CASE_ID_COL not in output_row or pd.isna(output_row[CASE_ID_COL]):
                if aggregation_level == "case":
                    raise ValueError(
                        "Inference input CSV must include a non-empty "
                        f"'{CASE_ID_COL}' column for slides missing from "
                        "project annotations when aggregation_level='case'."
                    )
                output_row[CASE_ID_COL] = slide_id

            synthesized.append(output_row)

        return pd.DataFrame(synthesized)

    def _create_inference_run_root(self) -> Path:
        if self.experiment.project_root is None:
            raise RuntimeError("experiment.project_root is not set.")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_root = (
            Path(self.experiment.project_root)
            / "inference"
            / f"{self.task_name}_{timestamp}"
        )
        suffix = 1
        unique_run_root = run_root
        while unique_run_root.exists():
            suffix += 1
            unique_run_root = run_root.with_name(f"{run_root.name}_{suffix}")

        unique_run_root.mkdir(parents=True, exist_ok=False)
        return unique_run_root

    def _write_invocation_inputs(
        self,
        *,
        inference_run_root: Path,
        input_csv: Path,
    ) -> None:
        shutil.copy2(input_csv, inference_run_root / "inference_input.csv")
        self.cfg.save_yaml(inference_run_root / "config_snapshot.yaml")

    def _prepare_datasets_for_combo(
        self,
        *,
        combo_cfg: ComboConfig,
        annotations_df: pd.DataFrame,
        input_annotations_df: pd.DataFrame,
    ) -> dict[str, list[Any]]:
        auto_bag_datasets = self._build_task_automatic_datasets(
            combo_cfg=combo_cfg,
            annotations_df=annotations_df,
        )
        input_bag_datasets = self._build_input_query_datasets(
            combo_cfg=combo_cfg,
            input_annotations_df=input_annotations_df,
        )
        datasets_by_use = group_datasets_by_use(auto_bag_datasets)
        input_use = self.task.get_inference_input_use()
        datasets_by_use.setdefault(input_use, []).extend(input_bag_datasets)
        return datasets_by_use

    def _build_task_automatic_datasets(
        self,
        *,
        combo_cfg: ComboConfig,
        annotations_df: pd.DataFrame,
    ) -> list[Any]:
        inference_dataset_uses = self.task.get_inference_dataset_uses()
        if inference_dataset_uses is None:
            selected_dataset_configs = [
                ds_cfg for ds_cfg in self.cfg.datasets if ds_cfg.used_for != "ignore"
            ]
        else:
            selected_dataset_configs = [
                ds_cfg
                for ds_cfg in self.cfg.datasets
                if ds_cfg.used_for in inference_dataset_uses
            ]

        bag_datasets: list[Any] = []
        for ds_cfg in selected_dataset_configs:
            dataset_annotations = annotations_df[
                annotations_df[DATASET_COL] == ds_cfg.name
            ].copy()
            if dataset_annotations.empty:
                logger.warning(
                    "[Inference] Skipping automatic dataset '%s' because no "
                    "annotation rows were found.",
                    ds_cfg.name,
                )
                continue

            self._ensure_features_exist_for_dataset(
                ds_cfg=ds_cfg,
                annotations_df=dataset_annotations,
                combo_cfg=combo_cfg,
            )
            bag_datasets.append(
                build_bag_dataset(
                    ds_cfg=ds_cfg,
                    annotations_df=dataset_annotations,
                    combo_cfg=combo_cfg,
                    aggregation_level=self.cfg.experiment.aggregation_level,
                    task=str(self.task_name),
                )
            )
        return bag_datasets

    def _build_input_query_datasets(
        self,
        *,
        combo_cfg: ComboConfig,
        input_annotations_df: pd.DataFrame,
    ) -> list[Any]:
        input_use = self.task.get_inference_input_use()
        dataset_config_by_name = {ds_cfg.name: ds_cfg for ds_cfg in self.cfg.datasets}
        bag_datasets: list[Any] = []

        for dataset_name, dataset_annotations in input_annotations_df.groupby(
            DATASET_COL,
            sort=False,
        ):
            source_ds_cfg = dataset_config_by_name[str(dataset_name)]
            inference_ds_cfg = source_ds_cfg.model_copy(update={"used_for": input_use})

            self._ensure_features_exist_for_dataset(
                ds_cfg=source_ds_cfg,
                annotations_df=dataset_annotations.copy(),
                combo_cfg=combo_cfg,
            )
            bag_datasets.append(
                build_bag_dataset(
                    ds_cfg=inference_ds_cfg,
                    annotations_df=dataset_annotations.copy(),
                    combo_cfg=combo_cfg,
                    aggregation_level=self.cfg.experiment.aggregation_level,
                    task=str(self.task_name),
                )
            )
        return bag_datasets

    def _ensure_features_exist_for_dataset(
        self,
        *,
        ds_cfg: DatasetEntry,
        annotations_df: pd.DataFrame,
        combo_cfg: ComboConfig,
    ) -> None:
        missing_slide_ids = find_slides_with_missing_features(
            ds_cfg=ds_cfg,
            annotations_df=annotations_df,
            combo_cfg=combo_cfg,
        )
        if not missing_slide_ids:
            return

        feature_dataset = self._build_feature_extraction_dataset(
            ds_cfg=ds_cfg,
            annotations_df=annotations_df,
            missing_slide_ids=missing_slide_ids,
            combo_cfg=combo_cfg,
        )
        self.feature_policy.execute_dataset(dataset=feature_dataset, combo_cfg=combo_cfg)

    def _build_feature_extraction_dataset(
        self,
        *,
        ds_cfg: DatasetEntry,
        annotations_df: pd.DataFrame,
        missing_slide_ids: list[str],
        combo_cfg: ComboConfig,
    ) -> WSIDataset:
        try:
            return build_wsi_dataset(
                ds_cfg=ds_cfg,
                annotations_df=annotations_df,
                slide_ids=missing_slide_ids,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                f"Cannot continue inference for dataset '{ds_cfg.name}' "
                f"(extractor={combo_cfg.feature_extraction}, "
                f"tile_px={combo_cfg.tile_px}, tile_mpp={combo_cfg.tile_mpp}) "
                "because some selected slides need feature extraction but are "
                f"not available locally. {error}"
            ) from error

    def _write_invocation_manifest(
        self,
        *,
        inference_run_root: Path,
        input_csv: Path,
        combinations: list[ComboConfig],
        task_outputs: list[dict[str, Any]],
    ) -> None:
        payload = {
            "task": self.task_name,
            "mode": "inference",
            "input_csv": str(Path(input_csv).resolve()),
            "num_combinations": len(combinations),
            "num_outputs": len(task_outputs),
            "combinations": [combo_cfg.to_dict() for combo_cfg in combinations],
            "task_outputs": task_outputs,
        }
        import json

        (inference_run_root / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
