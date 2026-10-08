# src/pathforge/core/datasets/wsi_dataset.py
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import logging
from math import isfinite
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from pathforge.config.config import DatasetEntry
from pathforge.core.datasets.base import DatasetBase
from pathforge.utils.constants import SLIDE_FILE_FORMATS

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WSI:
    """One whole-slide sample with source path, artifact path, and cached backend object."""

    slide: str
    patient: str
    category: str
    path: Path  # slide image path
    artifact_path: Path  # per-slide .h5 path
    fallback_mpp: Optional[float] = None

    _obj: Optional[Any] = field(default=None, repr=False)
    annotations: dict[str, Any] = field(default_factory=dict, kw_only=True)

    @classmethod
    def from_annotation(
        cls,
        row: Mapping[str, Any] | pd.Series,
        *,
        slide_path: str | Path,
        artifact_path: str | Path,
    ) -> WSI:
        """Build a WSI from one row without opening or discovering slide files.

        Missing patient/category values default to the slide ID/empty string.
        Invalid optional MPP values are ignored; typed fields are authoritative
        over the detached annotation copy.

        Example:
            ``WSI.from_annotation(row, slide_path=staged, artifact_path=h5_path)``.
        """
        annotations = deepcopy(dict(row))
        values: dict[str, str] = {}
        for key in ("slide", "patient", "category"):
            value = annotations.get(key)
            if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
                value = ""
            if not pd.api.types.is_scalar(value):
                raise ValueError(f"Annotation {key!r} must be a scalar value.")
            values[key] = str(value).strip()
        if not values["slide"]:
            raise ValueError("Annotation 'slide' must be a non-empty ID.")

        raw_mpp = annotations.get("fallback_mpp")
        fallback_mpp = None
        if raw_mpp is not None and not (
            pd.api.types.is_scalar(raw_mpp)
            and (pd.isna(raw_mpp) or str(raw_mpp).strip() == "")
        ):
            try:
                if not pd.api.types.is_scalar(raw_mpp) or pd.api.types.is_bool(raw_mpp):
                    raise ValueError("MPP must be a numeric scalar")
                parsed = float(raw_mpp)
                if not isfinite(parsed) or parsed <= 0:
                    raise ValueError("MPP must be positive and finite")
                fallback_mpp = parsed
            except (TypeError, ValueError, OverflowError):
                logger.warning(
                    "Invalid fallback_mpp for slide '%s': %r. Ignoring it.",
                    values["slide"],
                    raw_mpp,
                )

        return cls(
            slide=values["slide"],
            patient=values["patient"] or values["slide"],
            category=values["category"],
            path=Path(slide_path),
            artifact_path=Path(artifact_path),
            fallback_mpp=fallback_mpp,
            annotations=annotations,
        )

    @property
    def is_loaded(self) -> bool:
        return self._obj is not None

    @property
    def obj(self) -> Any:
        if self._obj is None:
            raise RuntimeError(
                "WSI not loaded. Backend must call processor.load_wsi(wsi) first."
            )
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
            logger.warning(
                "[%s] slides_dir does not exist or is not a directory: %s",
                self.name,
                self._slides_dir,
            )

        self._artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.samples = self._build_samples(annotations_df)

        logger.info(
            "[%s] Built %d WSI samples (used_for=%s)",
            self.name,
            len(self.samples),
            self.used_for,
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

        # ---- Exact direct-file match: <slide_id>.<ext> ----
        direct_matches = sorted(
            p for p in self._slides_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in SLIDE_FILE_FORMATS
            and p.stem == slide_id
        )

        # ---- Prefer direct files first ----
        if len(direct_matches) == 1:
            return direct_matches[0]

        if len(direct_matches) > 1:
            logger.warning(
                "[%s] Multiple exact direct-file matches found for slide '%s': %s. Skipping.",
                self.name,
                slide_id,
                [str(p) for p in direct_matches],
            )
            return None

        # ---- Only if no direct file exists, check exact DICOM-folder match ----
        slide_dir = self._slides_dir / slide_id
        if slide_dir.is_dir():
            dcm_matches = sorted(
                p for p in slide_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".dcm"
            )

            if dcm_matches:
                return dcm_matches[0]

            logger.warning(
                "[%s] Folder found for slide '%s' but no direct file match exists and the folder contains no .dcm files: %s",
                self.name,
                slide_id,
                slide_dir,
            )
            return None

        # ---- Not found ----
        logger.warning(
            "[%s] No valid slide source found for slide '%s' in '%s'. Expected either "
            "'<slide_id>.<ext>' or '<slide_id>/*.dcm'.",
            self.name,
            slide_id,
            self._slides_dir,
        )
        return None

    def _resolve_row_wsi_path(self, row: pd.Series) -> Optional[Path]:
        """Return a directly annotated WSI path when one is available."""
        if "wsi_path" not in row.index or pd.isna(row["wsi_path"]):
            return None

        candidate = Path(str(row["wsi_path"])).expanduser().resolve()
        if not candidate.exists():
            logger.warning(
                "[%s] Annotated wsi_path does not exist for slide '%s': %s",
                self.name,
                row.get("slide", "<unknown>"),
                candidate,
            )
            return None
        return candidate

    def _build_samples(self, ann_df: pd.DataFrame) -> list[WSI]:
        df = ann_df[ann_df["dataset"] == self.config.name]

        if df.empty:
            logger.warning(
                "[%s] No annotation rows found for this dataset name in the CSV.",
                self.name,
            )
            return []

        samples: list[WSI] = []

        for _, row in df.iterrows():
            slide_id = str(row["slide"])
            slide_path = self._resolve_row_wsi_path(row)
            if slide_path is None:
                slide_path = self._find_wsi_path(slide_id)
            if slide_path is None:
                continue

            samples.append(
                WSI.from_annotation(
                    row,
                    slide_path=slide_path,
                    artifact_path=self.slide_artifact_path(slide_id),
                )
            )

        if not samples:
            logger.warning(
                "[%s] No valid slides were found after scanning '%s'.",
                self.name,
                self._slides_dir,
            )

        return samples
