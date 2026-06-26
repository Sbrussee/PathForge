from __future__ import annotations

from pathlib import Path

import pandas as pd

from pathbench.core.evaluation.slide_retrieval.data import (
    SlideRetrievalEvaluationQuery,
)
from pathbench.utils.constants import (
    CASE_ID_COL,
    DATASET_COL,
    PATIENT_ID_COL,
    SLIDE_ID_COL,
)


def compute_dcg(relevance_values: list[int]) -> float:
    """Compute DCG for one ranked relevance list."""

    import numpy as np

    return float(
        sum(
            ((2 ** int(relevance)) - 1) / np.log2(rank + 1)
            for rank, relevance in enumerate(relevance_values, start=1)
        )
    )


def resolve_project_root_from_run_context(run_context: object) -> Path:
    """Find the project root by walking up from the run directory."""

    run_dir = Path(getattr(run_context, "run_dir"))
    for candidate in (run_dir, *run_dir.parents):
        if (candidate / "annotations.csv").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate project root with annotations.csv for run dir: {run_dir}"
    )


def resolve_id_column_for_aggregation(aggregation_level: str) -> str:
    """Map aggregation level to its corresponding annotation identifier column."""

    if aggregation_level == "slide":
        return SLIDE_ID_COL
    if aggregation_level == "case":
        return CASE_ID_COL
    if aggregation_level == "patient":
        return PATIENT_ID_COL
    raise ValueError(f"Unsupported aggregation level: {aggregation_level!r}")


def build_exclusion_key_from_row(
    row: pd.Series,
    *,
    aggregation_level: str,
    exclusion_level: str,
) -> str | None:
    """Recreate the retrieval exclusion key from aggregated annotation metadata."""

    if exclusion_level == "none":
        return None
    if exclusion_level == "slide":
        if aggregation_level != "slide":
            raise ValueError(
                "slide_retrieval.exclusion_level='slide' requires "
                "experiment.aggregation_level='slide'."
            )
        return str(row["sample_id"])
    if exclusion_level == "case":
        value = row.get(CASE_ID_COL)
        return None if pd.isna(value) or value is None else str(value)
    if exclusion_level == "patient":
        value = row.get(PATIENT_ID_COL)
        return None if pd.isna(value) or value is None else str(value)
    raise ValueError(f"Unsupported slide retrieval exclusion level: {exclusion_level!r}")


def build_aggregated_reference_pool(
    *,
    run_context: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build in-memory aggregated query/reference metadata from annotations.csv.

    Returns:
    - `all_items_df`: one row per aggregated sample across all datasets.
    - `reference_pool_df`: subset limited to datasets allowed as references.
    """

    project_root = resolve_project_root_from_run_context(run_context)
    annotations_path = project_root / "annotations.csv"
    annotations_df = pd.read_csv(annotations_path)

    aggregation_level = str(getattr(run_context, "aggregation_level"))
    label_column = str(getattr(run_context, "label_column"))
    manifest = dict(getattr(run_context, "manifest", {}))
    exclusion_level = str(manifest.get("exclusion_level", "patient"))
    id_column = resolve_id_column_for_aggregation(aggregation_level)
    reference_dataset_names = {
        str(name)
        for name in manifest.get("reference_dataset_names", [])
    }

    required_columns = {
        DATASET_COL,
        label_column,
        SLIDE_ID_COL,
        CASE_ID_COL,
        PATIENT_ID_COL,
    }
    missing_columns = sorted(
        column for column in required_columns if column not in annotations_df.columns
    )
    if missing_columns:
        raise ValueError(
            "annotations.csv is missing required columns for ndcg_at_k: "
            + ", ".join(missing_columns)
        )
    if id_column not in annotations_df.columns:
        raise ValueError(
            f"annotations.csv is missing aggregation id column '{id_column}' for ndcg_at_k."
        )

    working_df = annotations_df.copy()
    working_df = working_df[working_df[id_column].notna()].copy()
    working_df["sample_id"] = working_df[id_column].astype(str).str.strip()
    working_df = working_df[working_df["sample_id"] != ""].copy()
    working_df["_normalized_label"] = (
        working_df[label_column].astype(str).str.strip()
    )
    working_df = working_df[working_df["_normalized_label"] != ""].copy()

    def _aggregate_group(group_df: pd.DataFrame) -> pd.Series:
        normalized_labels = sorted(set(group_df["_normalized_label"].tolist()))
        if len(normalized_labels) != 1:
            raise ValueError(
                "ndcg_at_k requires exactly one label per aggregated sample. "
                f"Got sample_id={group_df.name!r} with labels={normalized_labels}"
            )

        def _single_optional_value(column_name: str) -> str | None:
            values = [
                str(value).strip()
                for value in group_df[column_name].tolist()
                if not pd.isna(value) and str(value).strip()
            ]
            unique_values = sorted(set(values))
            if not unique_values:
                return None
            if len(unique_values) != 1:
                raise ValueError(
                    "ndcg_at_k requires consistent aggregated identity metadata. "
                    f"Got sample_id={group_df.name!r} column={column_name!r} "
                    f"values={unique_values}"
                )
            return unique_values[0]

        dataset_values = sorted(
            {
                str(value).strip()
                for value in group_df[DATASET_COL].tolist()
                if not pd.isna(value) and str(value).strip()
            }
        )

        return pd.Series(
            {
                "sample_id": str(group_df.name),
                "label": normalized_labels[0],
                DATASET_COL: dataset_values[0] if len(dataset_values) == 1 else None,
                SLIDE_ID_COL: _single_optional_value(SLIDE_ID_COL),
                CASE_ID_COL: _single_optional_value(CASE_ID_COL),
                PATIENT_ID_COL: _single_optional_value(PATIENT_ID_COL),
            }
        )

    all_items_df = (
        working_df.groupby("sample_id", sort=True, dropna=False)
        .apply(_aggregate_group, include_groups=False)
        .reset_index(drop=True)
    )
    all_items_df["exclusion_key"] = all_items_df.apply(
        lambda row: build_exclusion_key_from_row(
            row,
            aggregation_level=aggregation_level,
            exclusion_level=exclusion_level,
        ),
        axis=1,
    )

    if not reference_dataset_names:
        raise ValueError(
            "ndcg_at_k could not determine any reference datasets from the run manifest."
        )
    reference_pool_df = all_items_df[
        all_items_df[DATASET_COL].isin(reference_dataset_names)
    ].copy()
    return all_items_df, reference_pool_df


def count_relevant_reference_items_for_query(
    *,
    query: SlideRetrievalEvaluationQuery,
    all_items_df: pd.DataFrame,
    reference_pool_df: pd.DataFrame,
) -> int:
    """Count findable same-label reference items for one query."""

    query_rows = all_items_df[all_items_df["sample_id"] == str(query.query_id)]
    if query_rows.empty:
        raise ValueError(
            f"ndcg_at_k could not resolve query metadata for sample_id={query.query_id!r}."
        )
    query_row = query_rows.iloc[0]
    query_exclusion_key = query_row["exclusion_key"]

    candidates_df = reference_pool_df[
        reference_pool_df["sample_id"] != str(query.query_id)
    ]
    if query_exclusion_key is not None:
        candidates_df = candidates_df[
            candidates_df["exclusion_key"] != query_exclusion_key
        ]

    relevant_candidates_df = candidates_df[
        candidates_df["label"] == str(query.query_label)
    ]
    return int(len(relevant_candidates_df))
