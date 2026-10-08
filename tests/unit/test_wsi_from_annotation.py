from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pathforge.core.datasets.wsi_dataset import WSI


def build(row):
    return WSI.from_annotation(row, slide_path="staged/renamed.svs", artifact_path="artifacts/S1.h5")


@pytest.mark.parametrize("row_type", [dict, pd.Series])
def test_annotation_constructor_copies_row_and_preserves_explicit_paths(row_type):
    row = row_type({"slide": "S1", "patient": 42, "category": "tumor", "fallback_mpp": "0.25", "wsi_path": "original/S1.svs", "custom": [1]})
    wsi = build(row)
    assert (wsi.slide, wsi.patient, wsi.category, wsi.fallback_mpp) == ("S1", "42", "tumor", 0.25)
    assert wsi.path == Path("staged/renamed.svs")
    assert wsi.artifact_path == Path("artifacts/S1.h5")
    row["custom"].append(2)
    assert wsi.annotations["custom"] == [1]
    wsi.annotations["fallback_mpp"] = 9
    assert wsi.fallback_mpp == 0.25
    assert not wsi.is_loaded


@pytest.mark.parametrize("value", [None, pd.NA, np.nan, "", "  ", "bad", 0, -1, np.inf, -np.inf, "inf", True, np.bool_(True), [0.5], {"mpp": 0.5}])
def test_missing_or_invalid_fallback_is_ignored(value):
    assert build({"slide": "S1", "fallback_mpp": value}).fallback_mpp is None


@pytest.mark.parametrize("value", [0.25, "0.5", np.float64(0.75)])
def test_valid_fallback_is_normalized(value):
    assert build({"slide": "S1", "fallback_mpp": value}).fallback_mpp == float(value)


@pytest.mark.parametrize("value", [None, pd.NA, np.nan, "", " "])
def test_missing_patient_and_category_use_defaults(value):
    wsi = build({"slide": "S1", "patient": value, "category": value})
    assert wsi.patient == "S1"
    assert wsi.category == ""
    assert wsi.fallback_mpp is None


@pytest.mark.parametrize("row", [{}, {"slide": None}, {"slide": pd.NA}, {"slide": " "}, {"slide": ["S1"]}])
def test_invalid_slide_id_raises(row):
    with pytest.raises(ValueError, match="slide"):
        build(row)


def test_direct_constructor_remains_compatible():
    obj = object()
    wsi = WSI("S1", "P1", "tumor", Path("a.svs"), Path("a.h5"), 0.5, obj)
    assert wsi.obj is obj
    assert wsi.fallback_mpp == 0.5
    other = WSI(slide="S2", patient="P2", category="", path=Path("b.svs"), artifact_path=Path("b.h5"))
    wsi.annotations["custom"] = "value"
    assert other.annotations == {}
    assert other.fallback_mpp is None
