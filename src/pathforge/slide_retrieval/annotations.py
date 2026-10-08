"""Resolve physical-slide annotations before retrieval reads source pixels."""

from __future__ import annotations

from typing import Any

import pandas as pd

from pathforge.core.datasets.factory import _resolve_slide_column


def resolve_sample_annotations(*, sample: Any, config: Any) -> dict[str, dict[str, Any]]:
    """Return member rows keyed by slide ID, selecting by dataset and slide.

    Prefer the sample's experiment snapshot; standalone callers load the
    configured annotation CSV once. Legacy callers without any annotation
    source get identity-only rows, preserving native-MPP processing.

    Example:
        ``rows = resolve_sample_annotations(sample=sample, config=cfg)``.
    """
    slide_ids = [str(value) for value in sample.slide_ids]
    frame = getattr(sample, "annotations_df", None)
    if frame is None:
        experiment = (
            config.get("experiment")
            if isinstance(config, dict)
            else getattr(config, "experiment", None)
        )
        annotation_file = (
            experiment.get("annotation_file")
            if isinstance(experiment, dict)
            else getattr(experiment, "annotation_file", None)
        )
        if annotation_file is None:
            return {slide_id: {"slide": slide_id} for slide_id in slide_ids}
        frame = pd.read_csv(annotation_file)

    dataset = dict(getattr(sample, "metadata", {}) or {}).get("dataset")
    if not dataset:
        raise ValueError("sample.metadata['dataset'] is required to resolve annotations.")
    if "dataset" not in frame.columns:
        raise ValueError("Annotations require dataset and slide columns.")
    slide_column = _resolve_slide_column(frame)
    dataset_rows = frame[frame["dataset"].astype(str) == str(dataset)]
    rows: dict[str, dict[str, Any]] = {}
    for slide_id in slide_ids:
        matches = dataset_rows[dataset_rows[slide_column].astype(str) == slide_id]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly 1 annotation row for dataset='{dataset}' "
                f"and slide='{slide_id}', but found {len(matches)}."
            )
        row = matches.iloc[0].to_dict()
        row["slide"] = slide_id
        rows[slide_id] = row
    return rows
