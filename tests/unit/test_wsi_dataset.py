# tests/unit/test_wsi_dataset.py

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from pathforge.config.config import DatasetEntry
from pathforge.core.datasets.wsi_dataset import WSIDataset
from pathforge.utils.constants import SLIDE_FILE_FORMATS


def _suffixes() -> list[str]:
    sufs = list(SLIDE_FILE_FORMATS)
    # make deterministic-ish for tests; also ensure they start with "."
    sufs = [s if s.startswith(".") else f".{s}" for s in sufs]
    return sorted(set(sufs))


def _make_ds(tmp_path: Path, *, name: str = "ds") -> tuple[WSIDataset, Path, Path]:
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = tmp_path / "artifacts"  # WSIDataset should create this
    ds_cfg = DatasetEntry(
        name=name,
        slides_dir=str(slides_dir),
        artifacts_dir=str(artifacts_dir),
        tissue_annotations_dir=None,
        used_for="all",
    )

    ann_df = pd.DataFrame(
        [{"dataset": name, "slide": "S1", "patient": "P1", "category": "C1"}]
    )

    ds = WSIDataset(ds_cfg, ann_df)
    return ds, slides_dir, artifacts_dir


def test_find_wsi_path_returns_none_when_missing(tmp_path: Path) -> None:
    ds, _, _ = _make_ds(tmp_path)
    assert ds._find_wsi_path("S1") is None


def test_find_wsi_path_finds_allowed_suffix(tmp_path: Path) -> None:
    ds, slides_dir, _ = _make_ds(tmp_path)
    suf = _suffixes()[0]

    p = slides_dir / f"S1{suf}"
    p.write_bytes(b"")

    found = ds._find_wsi_path("S1")
    assert found == p.resolve()


def test_find_wsi_path_ignores_disallowed_suffix(tmp_path: Path) -> None:
    ds, slides_dir, _ = _make_ds(tmp_path)

    (slides_dir / "S1.txt").write_text("not a slide", encoding="utf-8")

    found = ds._find_wsi_path("S1")
    assert found is None


def test_find_wsi_path_multiple_matches_returns_none(tmp_path: Path) -> None:
    ds, slides_dir, _ = _make_ds(tmp_path)

    sufs = _suffixes()
    suf1 = sufs[0]

    p1 = slides_dir / f"S1{suf1}"
    p1.write_bytes(b"")

    # If we have >=2 suffixes, create a second real variant; otherwise use case-variant.
    if len(sufs) >= 2:
        suf2 = sufs[1]
        p2 = slides_dir / f"S1{suf2}"
        p2.write_bytes(b"")
    else:
        p2 = slides_dir / f"S1{suf1.upper()}"
        p2.write_bytes(b"")

    found = ds._find_wsi_path("S1")
    assert found is None


def test_build_samples_filters_by_dataset_and_skips_missing_slides(
    tmp_path: Path,
) -> None:
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = tmp_path / "artifacts"  # should be created by WSIDataset

    suf = _suffixes()[0]

    # Existing slide for dataset "ds"
    (slides_dir / f"S1{suf}").write_bytes(b"")

    # Existing slide for another dataset (should be ignored by ds)
    (slides_dir / f"S3{suf}").write_bytes(b"")

    ds_cfg = DatasetEntry(
        name="ds",
        slides_dir=str(slides_dir),
        artifacts_dir=str(artifacts_dir),
        tissue_annotations_dir=None,
        used_for="all",
    )

    ann_df = pd.DataFrame(
        [
            {"dataset": "ds", "slide": "S1", "patient": "P1", "category": "C1"},  # kept
            {
                "dataset": "ds",
                "slide": "S2",
                "patient": "P2",
                "category": "C2",
            },  # missing -> skipped
            {
                "dataset": "other",
                "slide": "S3",
                "patient": "P3",
                "category": "C3",
            },  # ignored
        ]
    )

    ds = WSIDataset(ds_cfg, ann_df)

    # artifacts_dir should exist
    assert ds.artifacts_dir.is_dir()

    # Only S1 is a valid sample for this dataset
    assert len(ds.samples) == 1
    wsi = ds.samples[0]

    assert wsi.slide == "S1"
    assert wsi.patient == "P1"
    assert wsi.category == "C1"

    assert wsi.path == (slides_dir / f"S1{suf}").resolve()
    assert wsi.artifact_path == (ds.artifacts_dir / "S1.h5").resolve()


def test_build_samples_prefers_explicit_wsi_path_from_annotations(
    tmp_path: Path,
) -> None:
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = tmp_path / "artifacts"

    explicit_slide = tmp_path / "downloaded" / "remote_slide.svs"
    explicit_slide.parent.mkdir(parents=True, exist_ok=True)
    explicit_slide.write_bytes(b"")

    ds_cfg = DatasetEntry(
        name="remote_ds",
        slides_dir=str(slides_dir),
        artifacts_dir=str(artifacts_dir),
        tissue_annotations_dir=None,
        used_for="training",
    )

    ann_df = pd.DataFrame(
        [
            {
                "dataset": "remote_ds",
                "slide": "S1",
                "patient": "P1",
                "category": "C1",
                "wsi_path": str(explicit_slide),
            }
        ]
    )

    ds = WSIDataset(ds_cfg, ann_df)

    assert len(ds.samples) == 1
    assert ds.samples[0].path == explicit_slide.resolve()
