from __future__ import annotations

from pathlib import Path

import pandas as pd

from pathforge.config.config import DatasetEntry
from pathforge.core.experiments.combinations import ComboConfig
from pathforge.core.experiments.combo_ids import build_feature_name, build_tiling_id
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.core.io.slide_artifacts import features as features_io
from pathforge.utils.constants import DATASET_COL, SLIDE_ID_COL


def find_slides_with_missing_features(
    ds_cfg: DatasetEntry,
    annotations_df: pd.DataFrame,
    combo_cfg: ComboConfig,
) -> list[str]:
    """Return slide IDs for which the required features are missing."""
    dataset_df = annotations_df[annotations_df[DATASET_COL] == ds_cfg.name].copy()
    if dataset_df.empty:
        return []

    artifacts_dir = Path(ds_cfg.artifacts_dir).expanduser().resolve()

    slide_ids = (
        dataset_df[SLIDE_ID_COL]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    slide_ids_with_missing_features: list[str] = []
    tiling_id = build_tiling_id(combo_cfg)
    extractor_name = build_feature_name(combo_cfg)

    for slide_id in slide_ids:
        artifact_path = artifacts_dir / f"{slide_id}.h5"

        if not artifact_path.is_file():
            slide_ids_with_missing_features.append(slide_id)
            continue

        with FileHandleH5(artifact_path, mode="r") as slide_artifact:
            if not features_io.features_exist(
                slide_artifact,
                bag_id=tiling_id,
                extractor_name=extractor_name,
            ):
                slide_ids_with_missing_features.append(slide_id)

    return slide_ids_with_missing_features
