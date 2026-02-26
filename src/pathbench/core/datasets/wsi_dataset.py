# src/pathbench/core/datasets/wsi_dataset.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import logging
import pandas as pd

from pathbench.core.datasets.base import DatasetBase
from pathbench.config.config import DatasetEntry
from pathbench.utils.constants import SLIDE_FILE_FORMATS

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WSI:
    slide: str
    patient: str
    category: str
    path: Path  # slide image path
    artifact_path: Path  # per-slide .h5 path

    _obj: Optional[Any] = field(default=None, repr=False)

    @property
    def is_loaded(self) -> bool:
        return self._obj is not None

    @property
    def obj(self) -> Any:
        if self._obj is None:
            raise RuntimeError("WSI not loaded. Backend must call processor.load_wsi(wsi) first.")
        return self._obj


class WSIDataset(DatasetBase):
    """
    One sample = one WSI.
    Builds samples from annotations_df rows where ann_df['dataset'] == config.name.

    Artifacts are stored per-slide in:
      artifacts_dir/{slide_id}.h5

    Combos/tilings/features live inside that file (e.g. bags/{bag_id}/...),
    so this dataset does not track any active combo state.
    """

    def __init__(self, ds_cfg: DatasetEntry, annotations_df: pd.DataFrame):
        self._name = ds_cfg.name
        self.config = ds_cfg

        self._slides_dir = Path(ds_cfg.slides_dir).expanduser().resolve()
        self._artifacts_dir = Path(ds_cfg.artifacts_dir).expanduser().resolve()
        self._tissue_annotations_dir = (
            Path(ds_cfg.tissue_annotations_dir).expanduser().resolve()
            if ds_cfg.tissue_annotations_dir
            else None
        )

        logger.info(
            "Initializing WSIDataset '%s' slides_dir='%s' artifacts_dir='%s'",
            self.config.name,
            self._slides_dir,
            self._artifacts_dir,
        )

        if not self._slides_dir.is_dir():
            logger.warning("[%s] slides_dir does not exist or is not a directory: %s", self.name, self._slides_dir)

        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.samples = self._build_samples(annotations_df)

        logger.info("[%s] Built %d WSI samples (used_for=%s)", self.name, len(self.samples), self.used_for)

    # ---- DatasetBase API -------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def used_for(self) -> str:
        return self.config.used_for

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> WSI:
        return self.samples[idx]

    # ---- dirs / paths ----------------------------------------------------

    @property
    def slides_dir(self) -> Path:
        return self._slides_dir

    @property
    def artifacts_dir(self) -> Path:
        return self._artifacts_dir

    @property
    def tissue_annotations_dir(self) -> Optional[Path]:
        return self._tissue_annotations_dir

    def slide_artifact_path(self, slide_id: str) -> Path:
        """Per-slide HDF5 file path: artifacts_dir/{slide_id}.h5"""
        return self._artifacts_dir / f"{slide_id}.h5"

    # ---- internal helpers ------------------------------------------------

    def _find_wsi_path(self, slide_id: str) -> Optional[Path]:
        """Return the slide file path for slide_id, or None if not found."""
        pattern = f"{slide_id}.*"
        candidates = list(self._slides_dir.glob(pattern))

        candidates = [p for p in candidates if p.suffix.lower() in SLIDE_FILE_FORMATS]

        if not candidates:
            logger.warning(
                "[%s] No slide file found for slide '%s' in '%s' (pattern=%s)",
                self.name,
                slide_id,
                self._slides_dir,
                pattern,
            )
            return None

        if len(candidates) > 1:
            logger.warning(
                "[%s] Multiple files found for slide '%s': %s. Taking the first one.",
                self.name,
                slide_id,
                [str(p) for p in candidates],
            )

        return candidates[0]

    def _build_samples(self, ann_df: pd.DataFrame) -> list[WSI]:
        df = ann_df[ann_df["dataset"] == self.config.name]

        if df.empty:
            logger.warning("[%s] No annotation rows found for this dataset name in the CSV.", self.name)
            return []

        samples: list[WSI] = []

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            slide_id = str(row["slide"])
            patient = str(row["patient"])
            category = str(row["category"])

            logger.debug(
                "[%s] (%d/%d) slide_id='%s' (patient=%s, category=%s)",
                self.name,
                i,
                len(df),
                slide_id,
                patient,
                category,
            )

            slide_path = self._find_wsi_path(slide_id)
            if slide_path is None:
                continue

            samples.append(
                WSI(
                    slide=slide_id,
                    patient=patient,
                    category=category,
                    path=slide_path,
                    artifact_path=self.slide_artifact_path(slide_id),
                )
            )

        if not samples:
            logger.warning("[%s] No valid slides were found after scanning '%s'.", self.name, self._slides_dir)

        return samples
