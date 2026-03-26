from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Literal

import pandas as pd
import torch

from pathbench.config.config import DatasetEntry
from pathbench.core.datasets.base import BagDatasetBase
from pathbench.core.experiments.base import ComboConfig
from pathbench.core.experiments.combo_ids import build_tiling_id
from pathbench.core.io.h5.base import FileHandleH5
from pathbench.core.io.h5 import features as features_io
from pathbench.core.io.h5 import tiles as tiles_io
from pathbench.utils.constants import (
    AGGREGATION_LEVELS,
    CATEGORY_COL,
    SLIDE_ID_COL,
    CASE_ID_COL,
    PATIENT_ID_COL,
    DATASET_COL,
)


AggregationLevel = Literal[tuple(AGGREGATION_LEVELS)]
FeatureLevel = Literal["patch", "slide", "unknown", "invalid"]


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

        self.tiling_id = build_tiling_id(self.combo_cfg)
        self.extractor_name = str(self.combo_cfg.feature_extraction)
        self._feature_level: FeatureLevel = "unknown"

        df = self.annotations_df[self.annotations_df[DATASET_COL] == self.ds_cfg.name].copy()

        if df.empty:
            self.samples: list[BagSample] = []
            return

        self.samples = self._build_samples(df)
        self._feature_level = self._infer_feature_level()

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_bags(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any]:
        bag_tensor = self.load_bag(index)
        sample = self.samples[index]
        return bag_tensor, sample.category

    def get_sample(self, index: int) -> BagSample:
        return self.samples[index]

    def get_feature_level(self) -> FeatureLevel:
        """Return the inferred feature level for this dataset."""
        return self._feature_level

    def load_bag(self, index: int) -> torch.Tensor:
        """Load the bag tensor for one sample."""
        sample = self.samples[index]

        bags: list[torch.Tensor] = []
        for artifact_path in sample.artifact_paths:
            bag = self._load_slide_bag(artifact_path)
            bags.append(bag)

        if not bags:
            raise RuntimeError(f"No bags found for sample '{sample.sample_id}'.")

        return torch.cat(bags, dim=0).float()

    def get_bag_sample(self, index: int) -> tuple[torch.Tensor, BagSample]:
        """Return the bag tensor together with its BagSample."""
        return self.load_bag(index), self.get_sample(index)

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
                    bag_id=self.tiling_id,
                    extractor_name=self.extractor_name,
                ):
                    raise FileNotFoundError(
                        f"Missing features for sample '{sample_id}', slide '{slide_id}' in "
                        f"{artifact_path} for tiling_id='{self.tiling_id}', "
                        f"extractor='{self.extractor_name}'."
                    )

    # ------------------------------------------------------------------
    # Feature level
    # ------------------------------------------------------------------

    def _infer_feature_level(self, max_slides_to_check: int = 10) -> FeatureLevel:
        """
        Infer whether the stored features are patch-level or slide-level.

        Logic per slide artifact:
        - n_features == n_patches and n_patches > 1 -> patch
        - n_features == 1 and n_patches > 1 -> slide
        - n_features > 1 and n_features != n_patches -> invalid
        - n_features == 1 and n_patches == 1 -> ambiguous, continue checking

        If all checked slides are ambiguous, returns "unknown".
        If different non-ambiguous levels are encountered across slides,
        returns "invalid".
        """
        checked_paths: set[Path] = set()
        inferred_level: FeatureLevel | None = None
        checked_count = 0

        for sample in self.samples:
            for artifact_path in sample.artifact_paths:
                if artifact_path in checked_paths:
                    continue

                checked_paths.add(artifact_path)
                current_level = self._inspect_slide_feature_level(artifact_path)
                checked_count += 1

                if current_level == "invalid":
                    return "invalid"

                if current_level == "unknown":
                    if checked_count >= max_slides_to_check:
                        return inferred_level or "unknown"
                    continue

                if inferred_level is None:
                    inferred_level = current_level
                elif inferred_level != current_level:
                    return "invalid"

                if checked_count >= max_slides_to_check:
                    return inferred_level

        return inferred_level or "unknown"

    def _inspect_slide_feature_level(self, artifact_path: Path) -> FeatureLevel:
        """
        Infer feature level for one slide artifact.

        Returns:
            - "patch"
            - "slide"
            - "unknown"
            - "invalid"
        """
        with FileHandleH5(artifact_path, mode="r") as slide_artifact:
            try:
                feature_matrix = features_io.read_features(
                    slide_artifact,
                    bag_id=self.tiling_id,
                    extractor_name=self.extractor_name,
                )
                n_features = int(feature_matrix.shape[0])

                n_patches = tiles_io.coords_num_rows(
                    slide_artifact,
                    bag_id=self.tiling_id,
                )
            except Exception:
                return "invalid"

        if n_features <= 0 or n_patches <= 0:
            return "invalid"

        if n_features == n_patches:
            if n_patches > 1:
                return "patch"
            return "unknown"

        if n_features == 1 and n_patches > 1:
            return "slide"

        if n_features > 1 and n_features != n_patches:
            return "invalid"

        return "invalid"

    # ------------------------------------------------------------------
    # H5 loading
    # ------------------------------------------------------------------

    def _load_slide_bag(self, artifact_path: Path) -> torch.Tensor:
        with FileHandleH5(artifact_path, mode="r") as slide_artifact:
            feature_matrix = features_io.read_features(
                slide_artifact,
                bag_id=self.tiling_id,
                extractor_name=self.extractor_name,
            )

        return torch.from_numpy(feature_matrix)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _artifact_path(self, slide_id: str) -> Path:
        return self.artifacts_dir / f"{slide_id}.h5"