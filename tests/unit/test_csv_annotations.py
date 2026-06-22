from __future__ import annotations

from pathlib import Path

import pytest

from pathbench.core.annotations.csv import CSVAnnotations


def test_csv_annotations_loads_required_columns(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text(
        "slide,category,patient,dataset,wsi_path\n"
        "S1,tumor,P1,cohort_a,/tmp/slide1.svs\n",
        encoding="utf-8",
    )

    annotations = CSVAnnotations(str(annotation_path))

    assert annotations.contains_patient is True
    assert annotations.contains_dataset is True
    assert annotations.contains_wsi_path is True
    assert annotations.annotations["slide"] == ["S1"]
    assert annotations.annotations["dataset"] == ["cohort_a"]


def test_csv_annotations_fills_optional_columns_when_missing(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text(
        "slide,category\n"
        "S1,tumor\n"
        "S2,normal\n",
        encoding="utf-8",
    )

    annotations = CSVAnnotations(str(annotation_path))

    assert annotations.contains_patient is False
    assert annotations.contains_dataset is False
    assert annotations.contains_wsi_path is False
    assert annotations.annotations["patient"] == ["S1", "S2"]
    assert annotations.annotations["dataset"] == ["default", "default"]


def test_csv_annotations_requires_slide_and_category_columns(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text(
        "slide,label\n"
        "S1,tumor\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="category"):
        CSVAnnotations(str(annotation_path))
