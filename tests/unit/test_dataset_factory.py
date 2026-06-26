from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import pathforge.core.datasets.factory as factory_mod
from pathforge.core.datasets.factory import build_bag_dataset
from pathforge.core.experiments.combinations import ComboConfig


def _make_combo() -> ComboConfig:
    return ComboConfig(
        tile_px=256,
        tile_px_params={},
        tile_mpp=0.5,
        tile_mpp_params={},
        feature_extraction="uni",
        feature_extraction_params={},
    )


def _make_dataset_entry() -> SimpleNamespace:
    return SimpleNamespace(
        name="cohort_a",
        artifacts_dir="/tmp/artifacts",
        slides_dir="/tmp/slides",
        used_for="reference",
    )


def test_build_bag_dataset_routes_slide_retrieval_to_retrieval_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeRetrievalDataset:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    class _FakeMilDataset:
        def __init__(self, **kwargs) -> None:  # pragma: no cover
            raise AssertionError("MIL dataset constructor should not be used for slide retrieval.")

    monkeypatch.setattr(factory_mod, "SlideRetrievalBagDataset", _FakeRetrievalDataset)
    monkeypatch.setattr(factory_mod, "MILBagDataset", _FakeMilDataset)

    annotations_df = pd.DataFrame(
        {
            "dataset": ["cohort_a", "cohort_a", "cohort_b"],
            "slide": ["S1", "S2", "S9"],
            "category": [0, 1, 2],
        }
    )

    dataset = build_bag_dataset(
        ds_cfg=_make_dataset_entry(),
        annotations_df=annotations_df,
        combo_cfg=_make_combo(),
        aggregation_level="slide",
        task="slide_retrieval",
        target_column="category",
        slide_ids=["S2"],
    )

    assert isinstance(dataset, _FakeRetrievalDataset)
    assert captured["task"] == "slide_retrieval"
    assert captured["aggregation_level"] == "slide"
    assert captured["target_column"] == "category"
    filtered_annotations = captured["annotations_df"]
    assert isinstance(filtered_annotations, pd.DataFrame)
    assert filtered_annotations["slide"].tolist() == ["S2"]
    assert filtered_annotations["dataset"].tolist() == ["cohort_a"]


def test_build_bag_dataset_routes_mil_tasks_to_mil_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeRetrievalDataset:
        def __init__(self, **kwargs) -> None:  # pragma: no cover
            raise AssertionError("Retrieval dataset constructor should not be used for MIL tasks.")

    class _FakeMilDataset:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(factory_mod, "SlideRetrievalBagDataset", _FakeRetrievalDataset)
    monkeypatch.setattr(factory_mod, "MILBagDataset", _FakeMilDataset)

    annotations_df = pd.DataFrame(
        {
            "dataset": ["cohort_a", "cohort_a"],
            "slide_id": ["S1", "S2"],
            "category": [0, 1],
        }
    )

    dataset = build_bag_dataset(
        ds_cfg=_make_dataset_entry(),
        annotations_df=annotations_df,
        combo_cfg=_make_combo(),
        aggregation_level="patient",
        task="classification",
        target_column="category",
    )

    assert isinstance(dataset, _FakeMilDataset)
    assert captured["task"] == "classification"
    assert captured["aggregation_level"] == "patient"
    assert captured["target_column"] == "category"
    filtered_annotations = captured["annotations_df"]
    assert isinstance(filtered_annotations, pd.DataFrame)
    assert filtered_annotations["slide_id"].tolist() == ["S1", "S2"]
