from __future__ import annotations

import pandas as pd

from pathbench.core.datasets.bag_dataset import (
    BagDataset,
    MILBagDataset,
    SlideRetrievalBagDataset,
)
from pathbench.core.datasets.wsi_dataset import WSIDataset
from pathbench.config.config import Config, DatasetEntry
from pathbench.core.experiments.combinations import ComboConfig
from pathbench.utils.constants import DATASET_COL, SLIDE_ID_COL


def build_wsi_dataset(
    ds_cfg: DatasetEntry,
    annotations_df: pd.DataFrame,
    slide_ids: list[str] | None = None,
) -> WSIDataset:
    """Build a WSIDataset for one dataset, optionally restricted to specific slides."""
    dataset_annotations = annotations_df[
        annotations_df[DATASET_COL] == ds_cfg.name
    ].copy()

    if slide_ids is not None:
        slide_id_set = {str(slide_id) for slide_id in slide_ids}

        dataset_annotations = dataset_annotations[
            dataset_annotations[SLIDE_ID_COL].astype(str).isin(slide_id_set)
        ].copy()

        if dataset_annotations.empty:
            raise FileNotFoundError(
                f"No annotation rows found for requested slide_ids in dataset '{ds_cfg.name}'."
            )

    dataset = WSIDataset(ds_cfg, dataset_annotations)

    if slide_ids is not None:
        found_slide_ids = {wsi.slide for wsi in dataset.samples}
        missing_source_slide_ids = sorted(slide_id_set - found_slide_ids)

        if missing_source_slide_ids:
            raise FileNotFoundError(
                f"The following slides are not available in "
                f"slides_dir='{ds_cfg.slides_dir}': {missing_source_slide_ids}"
            )

    return dataset


def build_wsi_datasets(
    cfg: Config,
    annotations_df: pd.DataFrame,
) -> list[WSIDataset]:
    """Build WSIDataset objects for all non-ignored datasets."""
    datasets: list[WSIDataset] = []

    for ds_cfg in cfg.datasets:
        if ds_cfg.used_for == "ignore":
            continue

        datasets.append(build_wsi_dataset(ds_cfg, annotations_df))

    return datasets


def build_bag_dataset(
    ds_cfg: DatasetEntry,
    annotations_df: pd.DataFrame,
    combo_cfg: ComboConfig,
    aggregation_level: str,
    task: str,
    target_column: str | None = None,
    slide_ids: list[str] | None = None,
) -> BagDataset:
    """Build a BagDataset for one dataset, optionally restricted to specific slides."""
    dataset_annotations = annotations_df[
        annotations_df[DATASET_COL] == ds_cfg.name
    ].copy()

    if slide_ids is not None:
        slide_id_set = {str(slide_id) for slide_id in slide_ids}
        dataset_annotations = dataset_annotations[
            dataset_annotations[SLIDE_ID_COL].astype(str).isin(slide_id_set)
        ].copy()

        if dataset_annotations.empty:
            raise FileNotFoundError(
                f"No annotation rows found for requested slide_ids in dataset '{ds_cfg.name}'."
            )

    kwargs = {
        "ds_cfg": ds_cfg,
        "annotations_df": dataset_annotations,
        "combo_cfg": combo_cfg,
        "aggregation_level": aggregation_level,
    }

    if target_column is not None:
        kwargs["target_column"] = target_column

    dataset_cls: type[BagDataset]
    if str(task) == "slide_retrieval":
        dataset_cls = SlideRetrievalBagDataset
    else:
        dataset_cls = MILBagDataset

    return dataset_cls(**kwargs)


def build_bag_datasets(
    cfg: Config,
    annotations_df: pd.DataFrame,
    combo_cfg: ComboConfig,
    task: str,
    target_column: str | None = None,
) -> list[BagDataset]:
    """Build BagDataset objects for all non-ignored datasets."""
    datasets: list[BagDataset] = []

    for ds_cfg in cfg.datasets:
        if ds_cfg.used_for == "ignore":
            continue

        datasets.append(
            build_bag_dataset(
                ds_cfg=ds_cfg,
                annotations_df=annotations_df,
                combo_cfg=combo_cfg,
                aggregation_level=cfg.experiment.aggregation_level,
                task=task,
                target_column=target_column,
            )
        )

    return datasets
