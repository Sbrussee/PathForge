from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from pathforge.core.tasks.registry import get_task, import_task_modules
from pathforge.core.evaluation.base import TaskEvaluationAdapterBase
from pathforge.core.evaluation.registry import evaluation_task_adapter
from pathforge.core.evaluation.slide_retrieval.data import (
    SlideRetrievalEvaluationData,
    SlideRetrievalEvaluationHit,
    SlideRetrievalEvaluationQuery,
)
from pathforge.core.evaluation.slide_retrieval.pool import normalize_text_id
from pathforge.core.evaluation.types import EvaluationRunContext
from pathforge.core.experiments.combinations import ComboConfig, build_combinations
from pathforge.core.experiments.combo_ids import build_feature_name, build_tiling_id
from pathforge.slide_retrieval.io import (
    build_slide_retrieval_output_root,
    resolve_slide_retrieval_results_path,
)
from pathforge.utils.constants import CASE_ID_COL, PATIENT_ID_COL, SLIDE_ID_COL

_RANK_SAMPLE_PATTERN = re.compile(r"^rank_(?P<rank>[1-9]\d*)_sample_id$")


@evaluation_task_adapter("slide_retrieval")
class SlideRetrievalEvaluationAdapter(TaskEvaluationAdapterBase):
    """Evaluate saved slide-retrieval benchmark outputs."""

    task_name = "slide_retrieval"

    @classmethod
    def get_discovery_keys(cls) -> list[str]:
        import_task_modules()
        task_cls = get_task(cls.task_name)
        return task_cls.get_grid_keys()

    def discover_runs(self) -> list[EvaluationRunContext]:
        combos = build_combinations(
            cfg=self.cfg,
            keys=self.get_discovery_keys(),
        )
        label_column = self.cfg.evaluation.label_column
        if label_column is None:
            raise ValueError(
                "evaluation.label_column is required for slide-retrieval evaluation."
            )
        reference_dataset_names = tuple(
            sorted(
                str(ds_cfg.name)
                for ds_cfg in self.cfg.datasets
                if str(ds_cfg.used_for) in {"reference", "query_reference"}
            )
        )

        run_contexts: list[EvaluationRunContext] = []
        for combo_cfg in combos:
            search_root = build_slide_retrieval_output_root(
                project_root=str(self.experiment.project_root),
                tiling_id=build_tiling_id(combo_cfg),
                feature_name=build_feature_name(combo_cfg),
                slide_representation=str(combo_cfg.get("retrieval_representation")),
                search_method=str(combo_cfg.get("search_strategy")),
            )
            if not search_root.exists():
                continue

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
                if not self._manifest_matches_combo(
                    manifest=manifest, combo_cfg=combo_cfg
                ):
                    continue
                manifest.setdefault(
                    "reference_dataset_names", list(reference_dataset_names)
                )
                discovered_combo_cfg = self._resolve_run_combo_cfg(
                    manifest=manifest,
                    fallback_combo_cfg=combo_cfg,
                )
                run_contexts.append(
                    EvaluationRunContext(
                        task_name=self.task_name,
                        run_dir=run_dir.resolve(),
                        combo_cfg=discovered_combo_cfg,
                        manifest=manifest,
                        label_column=label_column,
                        aggregation_level=str(
                            manifest.get(
                                "aggregation_level",
                                self.cfg.experiment.aggregation_level,
                            )
                        ),
                    )
                )

        return run_contexts

    def load_run_data(
        self,
        run_context: EvaluationRunContext,
    ) -> SlideRetrievalEvaluationData:
        annotations_df = self.experiment.load_annotations()
        raw_results = self._load_raw_results(
            resolve_slide_retrieval_results_path(
                run_context.run_dir / "query_results.xlsx"
            )
        )
        label_lookup = self._build_label_lookup(
            annotations_df=annotations_df,
            sample_ids=self._collect_sample_ids(raw_results),
            aggregation_level=run_context.aggregation_level,
            label_column=run_context.label_column,
        )

        queries: list[SlideRetrievalEvaluationQuery] = []
        for raw_result in raw_results:
            query_id = normalize_text_id(raw_result["query_id"])
            hits = [
                SlideRetrievalEvaluationHit(
                    sample_id=normalize_text_id(hit["sample_id"]),
                    label=label_lookup[normalize_text_id(hit["sample_id"])],
                    score=float(hit["score"]),
                    rank=int(hit["rank"]),
                )
                for hit in raw_result["hits"]
            ]
            queries.append(
                SlideRetrievalEvaluationQuery(
                    query_id=query_id,
                    query_label=label_lookup[query_id],
                    hits=hits,
                )
            )

        return SlideRetrievalEvaluationData(queries=queries)

    def _load_raw_results(self, path: Path) -> list[dict[str, Any]]:
        if path.suffix.lower() == ".xlsx":
            results_df = pd.read_excel(path)
        else:
            results_df = pd.read_csv(path)
        if "query_sample_id" not in results_df.columns:
            raise ValueError(
                f"Slide-retrieval results file is missing required column 'query_sample_id': {path}"
            )
        raw_results: list[dict[str, Any]] = []
        for row_index, row in results_df.iterrows():
            query_id = normalize_text_id(row["query_sample_id"])
            if not query_id:
                raise ValueError(
                    "Slide-retrieval results contain an empty query_sample_id "
                    f"at row {row_index + 2} in {path}."
                )
            hits: list[dict[str, Any]] = []
            for column_name, value in row.items():
                match = _RANK_SAMPLE_PATTERN.fullmatch(str(column_name))
                if match is None or pd.isna(value):
                    continue

                rank = int(match.group("rank"))
                sample_id = normalize_text_id(value)
                if not sample_id:
                    raise ValueError(
                        "Slide-retrieval results contain an empty hit sample_id "
                        f"at row {row_index + 2}, rank {rank} in {path}."
                    )
                score_column = f"rank_{rank}_score"
                score_value = row.get(score_column)
                score = 0.0 if pd.isna(score_value) else float(score_value)
                hits.append(
                    {
                        "sample_id": sample_id,
                        "score": score,
                        "rank": rank,
                    }
                )

            raw_results.append(
                {
                    "query_id": query_id,
                    "hits": sorted(hits, key=lambda hit: int(hit["rank"])),
                }
            )

        return raw_results

    def _collect_sample_ids(self, raw_results: list[dict[str, Any]]) -> set[str]:
        sample_ids: set[str] = set()
        for raw_result in raw_results:
            sample_ids.add(str(raw_result["query_id"]))
            for hit in raw_result["hits"]:
                sample_ids.add(str(hit["sample_id"]))
        return sample_ids

    def _build_label_lookup(
        self,
        *,
        annotations_df: pd.DataFrame,
        sample_ids: set[str],
        aggregation_level: str,
        label_column: str,
    ) -> dict[str, str]:
        id_column = self._resolve_id_column(aggregation_level)
        required_columns = {id_column, label_column}
        missing_columns = sorted(required_columns - set(annotations_df.columns))
        if missing_columns:
            raise ValueError(
                "annotations.csv is missing required column(s) for "
                "slide-retrieval evaluation: " + ", ".join(missing_columns)
            )
        working_df = annotations_df.copy()
        working_df["_evaluation_sample_id"] = working_df[id_column].map(
            normalize_text_id
        )
        missing_labels: list[str] = []
        inconsistent_groups: list[str] = []
        label_lookup: dict[str, str] = {}

        for sample_id in sorted(sample_ids):
            normalized_sample_id = normalize_text_id(sample_id)
            matching_rows = working_df[
                working_df["_evaluation_sample_id"] == normalized_sample_id
            ]
            if matching_rows.empty:
                missing_labels.append(
                    f"{sample_id} (no rows found for column '{id_column}')"
                )
                continue

            label_series = matching_rows[label_column]
            missing_row_count = int(label_series.isna().sum())
            if missing_row_count > 0:
                missing_labels.append(
                    f"{sample_id} ({missing_row_count} rows have no '{label_column}')"
                )
                continue

            normalized_labels = {
                str(label).strip()
                for label in label_series.tolist()
                if str(label).strip()
            }
            if not normalized_labels:
                missing_labels.append(
                    f"{sample_id} (all '{label_column}' values are empty)"
                )
                continue

            if len(normalized_labels) != 1:
                inconsistent_groups.append(
                    f"{sample_id} -> {sorted(normalized_labels)}"
                )
                continue

            label_lookup[sample_id] = next(iter(normalized_labels))

        if missing_labels or inconsistent_groups:
            message_parts = [
                "Slide-retrieval evaluation label resolution failed.",
            ]
            if missing_labels:
                message_parts.append("Missing labels: " + "; ".join(missing_labels))
            if inconsistent_groups:
                message_parts.append(
                    "Inconsistent aggregated labels: " + "; ".join(inconsistent_groups)
                )
            raise ValueError(" ".join(message_parts))

        return label_lookup

    def _resolve_id_column(self, aggregation_level: str) -> str:
        if aggregation_level == "slide":
            return SLIDE_ID_COL
        if aggregation_level == "case":
            return CASE_ID_COL
        if aggregation_level == "patient":
            return PATIENT_ID_COL
        raise ValueError(
            f"Unsupported aggregation level for evaluation: {aggregation_level!r}"
        )

    def _manifest_matches_combo(
        self, *, manifest: dict[str, Any], combo_cfg: ComboConfig
    ) -> bool:
        """Return whether manifest-level identifiers match the searched combo."""
        expected_values = {
            "tiling_id": build_tiling_id(combo_cfg),
            "feature_extraction": build_feature_name(combo_cfg),
            "slide_representation": str(combo_cfg.get("retrieval_representation")),
            "search_method": str(combo_cfg.get("search_strategy")),
        }
        if not all(
            manifest.get(key) is None or str(manifest[key]) == str(expected_value)
            for key, expected_value in expected_values.items()
        ):
            return False
        manifest_combo_cfg = manifest.get("combo_cfg")
        if isinstance(manifest_combo_cfg, dict) and manifest_combo_cfg:
            return manifest_combo_cfg == combo_cfg.to_dict()

        return self._legacy_manifest_params_match_combo(
            manifest=manifest,
            combo_cfg=combo_cfg,
        )

    def _legacy_manifest_params_match_combo(
        self, *, manifest: dict[str, Any], combo_cfg: ComboConfig
    ) -> bool:
        """Match legacy method parameters without requiring unavailable full combo data.

        The legacy manifest only records resolved strategy parameters. Every
        parameter explicitly requested by the active combination must therefore
        agree with that record; extra resolved defaults are permitted.

        Example:
            A legacy manifest with ``{"radius": 4, "default": 1}`` matches a
            combo that explicitly requests ``{"radius": 4}``.
        """
        expected_params_by_manifest_key = {
            "slide_representation_params": combo_cfg.get_hyperparams(
                "retrieval_representation"
            ),
            "search_params": combo_cfg.get_hyperparams("search_strategy"),
        }
        return not any(
            isinstance(manifest.get(manifest_key), dict)
            and any(
                manifest[manifest_key].get(name) != value
                for name, value in expected_params.items()
            )
            for manifest_key, expected_params in expected_params_by_manifest_key.items()
        )

    def _resolve_run_combo_cfg(
        self, *, manifest: dict[str, Any], fallback_combo_cfg: ComboConfig
    ) -> ComboConfig:
        """Use persisted combo metadata when available, otherwise fall back."""
        manifest_combo_cfg = manifest.get("combo_cfg")
        if isinstance(manifest_combo_cfg, dict) and manifest_combo_cfg:
            return ComboConfig(**manifest_combo_cfg)
        return fallback_combo_cfg
