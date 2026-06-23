from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from pathbench.core.tasks.registry import get_task, import_task_modules
from pathbench.core.evaluation.base import TaskEvaluationAdapterBase
from pathbench.core.evaluation.registry import evaluation_task_adapter
from pathbench.core.evaluation.slide_retrieval.data import (
    SlideRetrievalEvaluationData,
    SlideRetrievalEvaluationHit,
    SlideRetrievalEvaluationQuery,
)
from pathbench.core.evaluation.types import EvaluationRunContext
from pathbench.core.experiments.combinations import build_combinations
from pathbench.core.experiments.combo_ids import build_feature_name, build_tiling_id
from pathbench.slide_retrieval.io import (
    build_slide_retrieval_output_root,
    resolve_slide_retrieval_results_path,
)
from pathbench.utils.constants import CASE_ID_COL, PATIENT_ID_COL, SLIDE_ID_COL

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
                manifest["reference_dataset_names"] = list(reference_dataset_names)
                run_contexts.append(
                    EvaluationRunContext(
                        task_name=self.task_name,
                        run_dir=run_dir.resolve(),
                        combo_cfg=combo_cfg,
                        manifest=manifest,
                        label_column=label_column,
                        aggregation_level=str(self.cfg.experiment.aggregation_level),
                    )
                )

        return run_contexts

    def load_run_data(
        self,
        run_context: EvaluationRunContext,
    ) -> SlideRetrievalEvaluationData:
        annotations_df = self.experiment.load_annotations()
        raw_results = self._load_raw_results(
            resolve_slide_retrieval_results_path(run_context.run_dir / "query_results.xlsx")
        )
        label_lookup = self._build_label_lookup(
            annotations_df=annotations_df,
            sample_ids=self._collect_sample_ids(raw_results),
            aggregation_level=run_context.aggregation_level,
            label_column=run_context.label_column,
        )

        queries: list[SlideRetrievalEvaluationQuery] = []
        for raw_result in raw_results:
            query_id = str(raw_result["query_id"])
            hits = [
                SlideRetrievalEvaluationHit(
                    sample_id=str(hit["sample_id"]),
                    label=label_lookup[str(hit["sample_id"])],
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
        raw_results: list[dict[str, Any]] = []
        for _, row in results_df.iterrows():
            query_id = str(row["query_sample_id"])
            hits: list[dict[str, Any]] = []
            for column_name, value in row.items():
                match = _RANK_SAMPLE_PATTERN.fullmatch(str(column_name))
                if match is None or pd.isna(value):
                    continue

                rank = int(match.group("rank"))
                score_column = f"rank_{rank}_score"
                score_value = row.get(score_column)
                score = 0.0 if pd.isna(score_value) else float(score_value)
                hits.append(
                    {
                        "sample_id": str(value),
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
        missing_labels: list[str] = []
        inconsistent_groups: list[str] = []
        label_lookup: dict[str, str] = {}

        for sample_id in sorted(sample_ids):
            matching_rows = annotations_df[annotations_df[id_column] == sample_id]
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
                message_parts.append(
                    "Missing labels: " + "; ".join(missing_labels)
                )
            if inconsistent_groups:
                message_parts.append(
                    "Inconsistent aggregated labels: "
                    + "; ".join(inconsistent_groups)
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
