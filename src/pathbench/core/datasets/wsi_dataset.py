# src/pathbench/core/datasets/wsi_dataset.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Any
import os
import glob
import logging

import pandas as pd

from pathbench.core.datasets.base import DatasetBase
from pathbench.config.config import DatasetEntry
from pathbench.utils.constants import SLIDE_FILE_FORMATS

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class WSI:
    # ---- identity / metadata ----
    slide: str
    patient: str
    category: str
    path: Path

    # ---- active combo context (valid for ONE combo only) ----
    active_combo: Optional[str] = None
    active_tissues_path: Optional[Path] = None
    active_tiles_path: Optional[Path] = None
    active_features_path: Optional[Path] = None

    # ---- backend-native object ----
    _obj: Optional[Any] = field(default=None, repr=False)

    # ---- active binding helpers ----
    def _assert_active_combo(self, combo: str) -> None:
        if self.active_combo is None:
            self.active_combo = combo
            return
        if self.active_combo != combo:
            raise RuntimeError(
                f"WSI '{self.slide}' has active_combo='{self.active_combo}', "
                f"but got combo='{combo}'. Reset before switching combos."
            )

    def clear_active(self) -> None:
        self.active_combo = None
        self.active_tissues_path = None
        self.active_tiles_path = None
        self.active_features_path = None

    def bind_active_tissues(self, combo: str, path: Path) -> None:
        self._assert_active_combo(combo)
        self.active_tissues_path = path

    def bind_active_tiles(self, combo: str, path: Path) -> None:
        self._assert_active_combo(combo)
        self.active_tiles_path = path

    def bind_active_features(self, combo: str, path: Path) -> None:
        self._assert_active_combo(combo)
        self.active_features_path = path

    # ---- loaded object convenience ----
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
    Dataset representing WSIs (one sample = one slide), built from:
    - annotations_df (CSV loaded into a DataFrame)
    - DatasetEntry config (slide_path, roi_path, tiles_path, features_path, used_for, ...)

    Adds an "active combo" mechanism:
    - dataset.active_combo tracks which combo is currently active
    - each WSI can store active_*_path fields valid only for that combo
    - switching combos resets active fields by default
    """

    def __init__(self, ds_cfg: DatasetEntry, annotations_df: pd.DataFrame):
        self._name = ds_cfg.name
        self.config = ds_cfg

        # Tracks which combo is currently active for this dataset (optional)
        self.active_combo: Optional[str] = None

        logger.info(
            "Initializing WSIDataset for dataset '%s' with slide_dir='%s'",
            self.config.name,
            self.config.slide_path,
        )
        logger.debug("[%s] Total rows in annotations_df: %d", self.config.name, len(annotations_df))

        self.samples: List[WSI] = self._build_samples(annotations_df)

        logger.info(
            "[%s] Built %d WSI samples (used_for=%s)",
            self.config.name,
            len(self.samples),
            self.config.used_for,
        )

    # ---- active combo management -----------------------------------------

    def set_active_combo(self, combo: str, *, reset: bool = True) -> None:
        """
        Set the dataset's active combo.

        If a different combo was already active:
        - reset=True (default): clears all slide active_* paths and switches
        - reset=False: raises
        """
        if self.active_combo == combo:
            return

        if self.active_combo is not None and self.active_combo != combo:
            if not reset:
                raise RuntimeError(
                    f"Dataset '{self.name}' already has active_combo='{self.active_combo}'. "
                    f"Refusing to switch to '{combo}' without reset."
                )
            self.reset_active()

        self.active_combo = combo

    def reset_active(self) -> None:
        """Clear dataset.active_combo and clear active_* fields for all slides."""
        self.active_combo = None
        for s in self.samples:
            s.clear_active()

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

    # ---- Convenience properties for base dirs ----------------------------

    @property
    def slide_dir(self) -> str:
        """Absolute path to the directory containing WSIs for this dataset."""
        return self.config.slide_path

    @property
    def rois_dir(self) -> str:
        """Absolute path where ROI geojson for this dataset should be stored."""
        return self.config.roi_path  # type: ignore[attr-defined]

    @property
    def tiles_dir(self) -> str:
        """Absolute path where tile index / npz records for this dataset should be stored."""
        return self.config.tiles_path  # type: ignore[attr-defined]

    @property
    def features_dir(self) -> str:
        """Absolute path where feature bags should be stored for this dataset."""
        return self.config.features_path  # type: ignore[attr-defined]

    # ---- internal helpers ------------------------------------------------

    def _find_wsi_path(self, slide_dir: str, slide_id: str) -> Optional[Path]:
        """Return the path to the slide file for this slide_id, or None if not found."""
        pattern = os.path.join(slide_dir, f"{slide_id}.*")
        logger.debug(
            "[%s] Looking for slide_id='%s' with pattern='%s'",
            self.config.name,
            slide_id,
            pattern,
        )

        candidates = glob.glob(pattern)
        logger.debug("[%s] Candidates for '%s': %s", self.config.name, slide_id, candidates)

        # keep only allowed extensions
        candidates = [
            p for p in candidates if os.path.splitext(p)[1].lower() in SLIDE_FILE_FORMATS
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

        return Path(candidates[0])

    def _build_samples(self, ann_df: pd.DataFrame) -> List[WSI]:
        # Filter annotations to this dataset
        df = ann_df[ann_df["dataset"] == self.config.name]

        logger.debug("[%s] Rows in annotations for this dataset: %d", self.config.name, len(df))

        slide_dir = self.config.slide_path
        if not os.path.isdir(slide_dir):
            logger.warning("[%s] slide_path '%s' does not exist or is not a directory.", self.config.name, slide_dir)

        samples: List[WSI] = []

        if df.empty:
            logger.warning("[%s] No annotation rows found for this dataset name in the CSV.", self.config.name)
            return samples

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            slide_id = row["slide"]
            patient = row["patient"]
            category = row["category"]

            logger.debug(
                "[%s] (%d/%d) Processing slide_id='%s' (patient=%s, category=%s)",
                self.config.name,
                i,
                len(df),
                slide_id,
                patient,
                category,
            )

            wsi_path = self._find_wsi_path(slide_dir, slide_id)
            if wsi_path is None:
                continue

            samples.append(
                WSI(
                    slide=slide_id,
                    patient=patient,
                    category=category,
                    path=wsi_path,
                )
            )

        if not samples:
            logger.warning("[%s] No valid slides were found after scanning '%s'.", self.config.name, slide_dir)

        return samples
