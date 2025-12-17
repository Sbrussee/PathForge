# src/pathbench/core/datasets/slides.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple, List
import os
import glob
import logging
import pandas as pd

from pathbench.core.datasets.base import DatasetBase
from pathbench.config.config import DatasetEntry
from pathbench.utils.constants import SLIDE_FILE_FORMATS

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SlideSample:
    slide: str
    patient: str
    category: str
    wsi_path: str


class SlideDataset(DatasetBase):
    """
    Dataset representing WSIs (one sample = one slide).
    Built from an annotations CSV + a DatasetEntry.
    """

    def __init__(self, ds_cfg: DatasetEntry, annotations_df: pd.DataFrame):
        self._name = ds_cfg.name
        self.config = ds_cfg  # carries slide_path, features_dir, tile_records_dir, used_for, ...

        logger.info(
            "Initializing SlideDataset for dataset '%s' with slide_dir='%s'",
            self.config.name,
            self.config.slide_path,
        )
        logger.debug(
            "[%s] Total rows in annotations_df: %d",
            self.config.name,
            len(annotations_df),
        )

        self.samples: List[SlideSample] = self._build_samples(annotations_df)

        logger.info(
            "[%s] Built %d slide samples (used_for=%s)",
            self.config.name,
            len(self.samples),
            self.config.used_for,
        )

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

    def __getitem__(self, idx: int) -> SlideSample:
        return self.samples[idx]

    # ---- New convenience properties for paths ----------------------------

    @property
    def slide_dir(self) -> str:
        """Absolute path to the directory containing WSIs for this dataset."""
        return self.config.slide_path

    @property
    def rois_dir(self) -> str:
        """
        Absolute path where ROI geojson for this dataset should be stored.
        """
        return self.config.roi_path  # type: ignore[attr-defined]
    
    @property
    def tiles_dir(self) -> str:
        """
        Absolute path where tile index / npz records for this dataset should be stored.
        """
        return self.config.tiles_path  # type: ignore[attr-defined]
    
    @property
    def features_dir(self) -> str:
        """
        Absolute path where feature bags (.pt) for this dataset should be stored.
        """
        return self.config.features_path  # type: ignore[attr-defined]

    # ---- internal helpers ------------------------------------------------

    def _find_wsi_path(self, slide_dir: str, slide_id: str) -> str | None:
        """Return the path to the slide file for this slide_id, or None if not found."""
        pattern = os.path.join(slide_dir, f"{slide_id}.*")
        logger.debug(
            "[%s] Looking for slide_id='%s' with pattern='%s'",
            self.config.name,
            slide_id,
            pattern,
        )

        candidates = glob.glob(pattern)
        logger.debug(
            "[%s] Candidates for '%s': %s",
            self.config.name,
            slide_id,
            candidates,
        )

        # keep only allowed extensions
        candidates = [
            path
            for path in candidates
            if os.path.splitext(path)[1].lower() in SLIDE_FILE_FORMATS
        ]

        if not candidates:
            logger.warning(
                "[%s] No slide file found for slide '%s' in '%s' (pattern=%s)",
                self.config.name,
                slide_id,
                slide_dir,
                pattern,
            )
            return None

        if len(candidates) > 1:
            logger.warning(
                "[%s] Multiple files found for slide '%s': %s. Taking the first one.",
                self.config.name,
                slide_id,
                candidates,
            )

        return candidates[0]

    def _build_samples(self, ann_df: pd.DataFrame) -> List[SlideSample]:
        # Filter annotations to this dataset
        df = ann_df[ann_df["dataset"] == self.config.name]

        logger.debug(
            "[%s] Rows in annotations for this dataset: %d",
            self.config.name,
            len(df),
        )

        slide_dir = self.config.slide_path
        if not os.path.isdir(slide_dir):
            logger.warning(
                "[%s] slide_path '%s' does not exist or is not a directory.",
                self.config.name,
                slide_dir,
            )

        samples: List[SlideSample] = []

        if df.empty:
            logger.warning(
                "[%s] No annotation rows found for this dataset name in the CSV.",
                self.config.name,
            )
            return samples

        for i, (_, row) in enumerate(df.iterrows()):
            slide_id = row["slide"]
            patient = row["patient"]
            category = row["category"]

            logger.debug(
                "[%s] (%d/%d) Processing slide_id='%s' (patient=%s, category=%s)",
                self.config.name,
                i + 1,
                len(df),
                slide_id,
                patient,
                category,
            )

            wsi_path = self._find_wsi_path(slide_dir, slide_id)
            if wsi_path is None:
                # _find_wsi_path already logs a warning
                continue

            logger.debug(
                "[%s] Matched slide_id='%s' -> '%s'",
                self.config.name,
                slide_id,
                wsi_path,
            )

            samples.append(
                SlideSample(
                    slide=slide_id,
                    patient=patient,
                    category=category,
                    wsi_path=wsi_path,
                )
            )

        if not samples:
            logger.warning(
                "[%s] No valid slides were found after scanning '%s'.",
                self.config.name,
                slide_dir,
            )

        return samples
