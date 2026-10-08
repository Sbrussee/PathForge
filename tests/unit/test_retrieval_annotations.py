from types import SimpleNamespace

import pandas as pd
import pytest

from pathforge.slide_retrieval.annotations import resolve_sample_annotations


def sample(frame=None):
    return SimpleNamespace(slide_ids=["S1", "S2"], metadata={"dataset": "a"}, annotations_df=frame)


def test_resolves_physical_member_rows_by_dataset_from_snapshot(monkeypatch):
    frame = pd.DataFrame([
        {"dataset": "a", "slide": "S1", "fallback_mpp": 0.25},
        {"dataset": "b", "slide": "S1", "fallback_mpp": 9},
        {"dataset": "a", "slide": "S2", "fallback_mpp": 0.5},
    ])
    monkeypatch.setattr(pd, "read_csv", lambda *_: pytest.fail("Snapshot must take precedence"))
    rows = resolve_sample_annotations(sample=sample(frame), config={"experiment": {"annotation_file": "unused.csv"}})
    assert rows["S1"]["fallback_mpp"] == 0.25
    assert rows["S2"]["fallback_mpp"] == 0.5


@pytest.mark.parametrize("count", [0, 2])
def test_missing_or_duplicate_rows_raise(count):
    frame = pd.DataFrame([{"dataset": "a", "slide": "S1"}] * count, columns=["dataset", "slide"])
    with pytest.raises(ValueError, match=f"dataset='a'.*slide='S1'.*found {count}"):
        resolve_sample_annotations(sample=sample(frame), config={})


def test_csv_is_loaded_once_for_all_members(tmp_path, monkeypatch):
    path = tmp_path / "annotations.csv"
    path.write_text("dataset,slide,fallback_mpp\na,S1,0.25\na,S2,0.5\nb,S1,9\n")
    read_csv = pd.read_csv
    calls = []
    monkeypatch.setattr(pd, "read_csv", lambda path: calls.append(path) or read_csv(path))
    rows = resolve_sample_annotations(sample=sample(), config=SimpleNamespace(experiment=SimpleNamespace(annotation_file=path)))
    assert len(calls) == 1
    assert rows["S1"]["fallback_mpp"] == 0.25


def test_legacy_native_mpp_callers_without_annotation_source():
    assert resolve_sample_annotations(sample=sample(), config={}) == {"S1": {"slide": "S1"}, "S2": {"slide": "S2"}}


def test_missing_dataset_or_columns_raise():
    value = sample(pd.DataFrame([{"slide": "S1"}]))
    with pytest.raises(ValueError, match="columns"):
        resolve_sample_annotations(sample=value, config={})
    value.metadata = {}
    with pytest.raises(ValueError, match="metadata"):
        resolve_sample_annotations(sample=value, config={})
