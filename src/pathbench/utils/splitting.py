from __future__ import annotations

"""Utilities for constructing leakage-free patient-level data splits."""

from typing import Any, Dict, Mapping, Sequence

import pandas as pd

from pathbench.config.config import Config


def build_patient_splits(
    annotations: pd.DataFrame,
    config: Config,
    dataset_sources: Sequence[Any],
) -> Dict[str, pd.DataFrame]:
    """Create train/validation/test splits without patient leakage.

    Splits are built from annotation rows and respect optional ``dataset``
    assignments provided via ``dataset_sources`` (``used_for`` attribute).
    Remaining validation data are sampled from the training pool at the patient
    level with configurable stratification.
    """

    patient_col = config.experiment.patient_column
    label_col = config.experiment.label_column
    center_col = config.experiment.center_column

    if patient_col not in annotations.columns:
        raise ValueError(
            f"Patient column '{patient_col}' is required to avoid patient leakage."
        )

    usage_map = _collect_dataset_usage(dataset_sources)
    dataset_available = "dataset" in annotations.columns and any(usage_map.values())

    splits: Dict[str, pd.DataFrame] = {"training": pd.DataFrame(), "validation": pd.DataFrame(), "testing": pd.DataFrame()}

    if dataset_available:
        for usage, names in usage_map.items():
            if names:
                splits[usage] = annotations[annotations["dataset"].isin(names)].copy()

    # fall back to all annotations for training when nothing assigned
    if splits["training"].empty:
        splits["training"] = annotations.copy()

    # Remove patients already assigned to held-out splits from training pool
    held_out_patients = _unique_patients(pd.concat([splits["validation"], splits["testing"]], ignore_index=True), patient_col)
    if held_out_patients:
        splits["training"] = splits["training"][~splits["training"][patient_col].isin(held_out_patients)].copy()

    # Derive validation split if none provided
    if splits["validation"].empty:
        train_df, val_df = _split_train_validation(
            splits["training"],
            patient_col=patient_col,
            stratify_by=config.experiment.stratify_by,
            stratify_column=_resolve_stratify_column(
                annotations,
                stratify_by=config.experiment.stratify_by,
                label_column=label_col,
                center_column=center_col,
                patient_column=patient_col,
            ),
            val_fraction=config.experiment.val_fraction,
            seed=config.experiment.split_seed,
        )
        splits["training"], splits["validation"] = train_df, val_df

    _ensure_no_patient_overlap(splits, patient_col)
    return splits


def _collect_dataset_usage(dataset_sources: Sequence[Any]) -> Dict[str, set[str]]:
    usage_map: Dict[str, set[str]] = {"training": set(), "validation": set(), "testing": set()}
    for ds in dataset_sources:
        usage = getattr(ds, "used_for", None)
        name = getattr(ds, "name", None)
        if usage in usage_map and name:
            usage_map[usage].add(name)
    return usage_map


def _resolve_stratify_column(
    annotations: pd.DataFrame,
    stratify_by: str,
    label_column: str,
    center_column: str | None,
    patient_column: str,
) -> str | None:
    if stratify_by == "label":
        if label_column not in annotations.columns:
            raise ValueError(f"Label column '{label_column}' missing for stratified split.")
        return label_column
    if stratify_by == "center":
        if center_column is None or center_column not in annotations.columns:
            raise ValueError("Center stratification requested but center column missing.")
        return center_column
    if stratify_by == "patient":
        return patient_column
    return None


def _split_train_validation(
    train_df: pd.DataFrame,
    patient_col: str,
    stratify_by: str,
    stratify_column: str | None,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if val_fraction <= 0 or train_df.empty:
        return train_df, pd.DataFrame(columns=train_df.columns)

    patient_table = _build_patient_table(train_df, patient_col, stratify_column)

    train_patients: list[str] = []
    val_patients: list[str] = []

    if stratify_column is None or stratify_by == "none":
        shuffled = patient_table.sample(frac=1.0, random_state=seed)
        split_idx = max(1, int(len(shuffled) * val_fraction)) if len(shuffled) > 1 else int(len(shuffled) * val_fraction)
        val_patients = shuffled.iloc[:split_idx][patient_col].tolist()
        train_patients = shuffled.iloc[split_idx:][patient_col].tolist()
    else:
        for _, group in patient_table.groupby(stratify_column):
            order = group.sample(frac=1.0, random_state=seed)
            split_idx = max(1, int(len(order) * val_fraction)) if len(order) > 1 else int(len(order) * val_fraction)
            val_patients.extend(order.iloc[:split_idx][patient_col].tolist())
            train_patients.extend(order.iloc[split_idx:][patient_col].tolist())

    train_subset = train_df[train_df[patient_col].isin(train_patients)].copy()
    val_subset = train_df[train_df[patient_col].isin(val_patients)].copy()
    return train_subset, val_subset


def _build_patient_table(df: pd.DataFrame, patient_col: str, stratify_col: str | None) -> pd.DataFrame:
    group_cols = [patient_col]
    if stratify_col is not None and stratify_col != patient_col:
        group_cols.append(stratify_col)
    aggregated = df[group_cols].drop_duplicates(subset=patient_col)
    if aggregated[patient_col].duplicated().any():
        raise ValueError("Patient identifiers must map to a single stratification value.")
    return aggregated


def _ensure_no_patient_overlap(splits: Mapping[str, pd.DataFrame], patient_col: str) -> None:
    seen: Dict[str, str] = {}
    for usage, df in splits.items():
        for patient in _unique_patients(df, patient_col):
            if patient in seen:
                raise ValueError(
                    f"Patient '{patient}' appears in both '{seen[patient]}' and '{usage}' splits, violating leakage constraints."
                )
            seen[patient] = usage


def _unique_patients(df: pd.DataFrame, patient_col: str) -> set[str]:
    if df.empty:
        return set()
    return set(df[patient_col].astype(str).unique())