from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Literal

import pandas as pd
import torch

from pathbench.config.config import DatasetEntry
from pathbench.core.datasets.base import BagDatasetBase
from pathbench.core.experiments.base import ComboConfig
from pathbench.core.io.h5.base import FileHandleH5
from pathbench.core.io.h5 import features as features_io
from pathbench.utils.constants import (
    AGGREGATION_LEVELS,
    CATEGORY_COL,
    SLIDE_ID_COL,
    CASE_ID_COL,
    PATIENT_ID_COL,
    DATASET_COL,
)


AggregationLevel = Literal[tuple(AGGREGATION_LEVELS)]


@dataclass(slots=True)
class BagSample:
    sample_id: str
    slide_ids: list[str]
    artifact_paths: list[Path]
    category: Any
    patient_id: Optional[str] = None
    case_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BagDataset(BagDatasetBase):
    """
    H5-backed bag dataset.

    Notes:
    - One BagDataset is built for one configured dataset.
    - Grouping happens in __post_init__ based on aggregation_level.
    - Features are loaded lazily in __getitem__.
    - Feature existence is checked eagerly while building samples.
    - Missing-feature extraction should be handled outside this class.
    """

    ds_cfg: DatasetEntry
    annotations_df: pd.DataFrame
    combo_cfg: ComboConfig
    aggregation_level: AggregationLevel = "slide"
    target_column: Optional[str] = CATEGORY_COL

    def __post_init__(self) -> None:
        self._name = self.ds_cfg.name
        self.artifacts_dir = Path(self.ds_cfg.artifacts_dir).expanduser().resolve()

        self.bag_id = self._build_bag_id(
            tile_px=int(self.combo_cfg.tile_px),
            tile_mpp=float(self.combo_cfg.tile_mpp),
        )
        self.extractor_name = str(self.combo_cfg.feature_extraction)

        df = self.annotations_df[self.annotations_df[DATASET_COL] == self.ds_cfg.name].copy()

        if df.empty:
            self.samples: list[BagSample] = []
            return

        self.samples = self._build_samples(df)

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_bags(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any]:
        sample = self.samples[index]

        bags: list[torch.Tensor] = []
        for artifact_path in sample.artifact_paths:
            bag = self._load_slide_bag(artifact_path)
            bags.append(bag)

        if not bags:
            raise RuntimeError(f"No bags found for sample '{sample.sample_id}'.")

        bag_tensor = torch.cat(bags, dim=0).float()
        return bag_tensor, sample.category

    def get_sample(self, index: int) -> BagSample:
        return self.samples[index]

    # ------------------------------------------------------------------
    # Sample building
    # ------------------------------------------------------------------

    def _build_samples(self, df: pd.DataFrame) -> list[BagSample]:
        if self.aggregation_level == "slide":
            return self._build_slide_samples(df)

        if self.aggregation_level == "case":
            if CASE_ID_COL not in df.columns:
                raise ValueError(
                    f"aggregation_level='case' requires column '{CASE_ID_COL}' in annotations."
                )
            return self._build_grouped_samples(df, group_col=CASE_ID_COL)

        if self.aggregation_level == "patient":
            if PATIENT_ID_COL not in df.columns:
                raise ValueError(
                    f"aggregation_level='patient' requires column '{PATIENT_ID_COL}' in annotations."
                )
            return self._build_grouped_samples(df, group_col=PATIENT_ID_COL)

        raise ValueError(f"Unsupported aggregation_level: {self.aggregation_level!r}")

    def _build_slide_samples(self, df: pd.DataFrame) -> list[BagSample]:
        samples: list[BagSample] = []

        for _, row in df.iterrows():
            slide_id = str(row[SLIDE_ID_COL])
            row_df = pd.DataFrame([row])
            artifact_paths = [self._artifact_path(slide_id)]

            self._ensure_features_exist(
                sample_id=slide_id,
                slide_ids=[slide_id],
                artifact_paths=artifact_paths,
            )

            samples.append(
                BagSample(
                    sample_id=slide_id,
                    slide_ids=[slide_id],
                    artifact_paths=artifact_paths,
                    category=self._resolve_category(row_df),
                    patient_id=self._resolve_single_optional_value(row_df, PATIENT_ID_COL),
                    case_id=self._resolve_single_optional_value(row_df, CASE_ID_COL),
                    metadata=self._build_metadata(row_df),
                )
            )

        return samples

    def _build_grouped_samples(self, df: pd.DataFrame, group_col: str) -> list[BagSample]:
        samples: list[BagSample] = []

        for group_value, group_df in df.groupby(group_col, sort=False):
            group_df = group_df.sort_values(SLIDE_ID_COL)
            slide_ids = [str(x) for x in group_df[SLIDE_ID_COL].tolist()]
            artifact_paths = [self._artifact_path(slide_id) for slide_id in slide_ids]

            self._ensure_features_exist(
                sample_id=str(group_value),
                slide_ids=slide_ids,
                artifact_paths=artifact_paths,
            )

            samples.append(
                BagSample(
                    sample_id=str(group_value),
                    slide_ids=slide_ids,
                    artifact_paths=artifact_paths,
                    category=self._resolve_category(group_df),
                    patient_id=self._resolve_single_optional_value(group_df, PATIENT_ID_COL),
                    case_id=self._resolve_single_optional_value(group_df, CASE_ID_COL),
                    metadata=self._build_metadata(group_df),
                )
            )

        return samples

    def _resolve_category(self, group_df: pd.DataFrame) -> Any:
        if self.target_column is None:
            return None

        if self.target_column not in group_df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in annotations.")

        values = group_df[self.target_column].dropna().unique().tolist()

        if len(values) == 0:
            return None

        if len(values) > 1:
            raise ValueError(
                f"Inconsistent target values for grouped bag in column "
                f"'{self.target_column}': {values}"
            )

        return values[0]

    def _resolve_single_optional_value(self, group_df: pd.DataFrame, column: str) -> Optional[str]:
        if column not in group_df.columns:
            return None

        values = group_df[column].dropna().astype(str).unique().tolist()

        if len(values) == 0:
            return None

        if len(values) > 1:
            raise ValueError(
                f"Inconsistent values for grouped bag in column '{column}': {values}"
            )

        return values[0]

    def _build_metadata(self, group_df: pd.DataFrame) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            DATASET_COL: self.ds_cfg.name,
            "aggregation_level": self.aggregation_level,
            "num_slides": len(group_df),
            "slide_ids": [str(x) for x in group_df[SLIDE_ID_COL].tolist()],
        }

        for col in (PATIENT_ID_COL, CASE_ID_COL, CATEGORY_COL):
            if col in group_df.columns:
                values = group_df[col].dropna().unique().tolist()
                metadata[col] = values[0] if len(values) == 1 else values

        return metadata

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    def _ensure_features_exist(
        self,
        *,
        sample_id: str,
        slide_ids: list[str],
        artifact_paths: list[Path],
    ) -> None:
        for slide_id, artifact_path in zip(slide_ids, artifact_paths):
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    f"Artifact file not found for sample '{sample_id}', slide '{slide_id}': "
                    f"{artifact_path}"
                )

            with FileHandleH5(artifact_path, mode="r") as slide_artifact:
                if not features_io.features_exist(
                    slide_artifact,
                    bag_id=self.bag_id,
                    extractor_name=self.extractor_name,
                ):
                    raise FileNotFoundError(
                        f"Missing features for sample '{sample_id}', slide '{slide_id}' in "
                        f"{artifact_path} for bag_id='{self.bag_id}', "
                        f"extractor='{self.extractor_name}'."
                    )

    # ------------------------------------------------------------------
    # H5 loading
    # ------------------------------------------------------------------

    def _load_slide_bag(self, artifact_path: Path) -> torch.Tensor:
        with FileHandleH5(artifact_path, mode="r") as slide_artifact:
            feature_matrix = features_io.read_features(
                slide_artifact,
                bag_id=self.bag_id,
                extractor_name=self.extractor_name,
            )

        return torch.from_numpy(feature_matrix)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _artifact_path(self, slide_id: str) -> Path:
        return self.artifacts_dir / f"{slide_id}.h5"

    @staticmethod
    def _build_bag_id(*, tile_px: int, tile_mpp: float) -> str:
        return f"{tile_px}px_{tile_mpp:g}mpp"