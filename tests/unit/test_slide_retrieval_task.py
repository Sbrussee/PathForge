from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pathbench.benchmarking.tasks.slide_retrieval import SlideRetrievalTask
import pathbench.benchmarking.tasks.slide_retrieval as slide_retrieval_task_module
from pathbench.core.experiments.base import ComboConfig
from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathbench.slide_retrieval.types import RetrievalItemMetadata


class _FakeBagDataset:
    def __init__(self, *, tiling_id: str, aggregation_level: str, num_bags: int = 0) -> None:
        self.tiling_id = tiling_id
        self.aggregation_level = aggregation_level
        self.num_bags = num_bags

    def get_bag_sample(self, index: int) -> tuple[object, object]:
        raise AssertionError(f"Unexpected get_bag_sample call for index {index}")


def _make_task(tmp_path: Path) -> SlideRetrievalTask:
    cfg = SimpleNamespace(experiment=SimpleNamespace())
    experiment = SimpleNamespace(cfg=cfg, project_root=str(tmp_path))
    return SlideRetrievalTask(experiment)


def test_infer_dataset_context_reads_shared_bag_context(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    datasets_by_use = {
        "reference": [_FakeBagDataset(tiling_id="bag_a", aggregation_level="slide")],
        "query": [_FakeBagDataset(tiling_id="bag_a", aggregation_level="slide")],
    }

    bag_id, aggregation_level = task._infer_dataset_context(
        datasets_by_use=datasets_by_use
    )

    assert bag_id == "bag_a"
    assert aggregation_level == "slide"


def test_infer_dataset_context_rejects_mixed_bag_ids(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    datasets_by_use = {
        "reference": [_FakeBagDataset(tiling_id="bag_a", aggregation_level="slide")],
        "query": [_FakeBagDataset(tiling_id="bag_b", aggregation_level="slide")],
    }

    with pytest.raises(ValueError, match="same bag_id"):
        task._infer_dataset_context(datasets_by_use=datasets_by_use)


def test_split_representations_by_use_supports_query_reference(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    reference_only = RetrievalRepresentation(
        sample_id="ref",
        representation_type="pooled",
        data=[1.0],
    )
    shared_item = RetrievalRepresentation(
        sample_id="shared",
        representation_type="pooled",
        data=[2.0],
    )
    query_only = RetrievalRepresentation(
        sample_id="query",
        representation_type="pooled",
        data=[3.0],
    )

    reference_items, query_items = task._split_representations_by_use(
        {
            "reference": [reference_only],
            "query_reference": [shared_item],
            "query": [query_only],
        }
    )

    assert reference_items == [reference_only, shared_item]
    assert query_items == [query_only, shared_item]


def test_ensure_representations_rejects_invalid_use(tmp_path: Path) -> None:
    task = _make_task(tmp_path)

    with pytest.raises(ValueError, match="Unsupported retrieval dataset use 'training'"):
        task._ensure_representations(
            datasets_by_use={
                "training": [_FakeBagDataset(tiling_id="bag_a", aggregation_level="slide")]
            },
            combo_cfg=SimpleNamespace(),
        )


def test_build_representation_metadata_merges_sample_and_strategy_metadata(
    tmp_path: Path,
) -> None:
    task = _make_task(tmp_path)
    sample = SimpleNamespace(
        category="tumor",
        patient_id="patient-1",
        case_id="case-1",
        slide_ids=["slide-1", "slide-2"],
        metadata={"source_dataset": "train"},
    )

    metadata = task._build_representation_metadata(
        sample=sample,
        strategy_metadata=RetrievalItemMetadata(
            center_id="center-a",
            extra={"strategy_tag": "pooled"},
        ),
    )

    assert metadata.category == "tumor"
    assert metadata.patient_id == "patient-1"
    assert metadata.case_id == "case-1"
    assert metadata.member_ids == ["slide-1", "slide-2"]
    assert metadata["source_dataset"] == "train"
    assert metadata.center_id == "center-a"
    assert metadata["strategy_tag"] == "pooled"


def test_execute_reads_strategy_hyperparams_from_combo_cfg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _make_task(tmp_path)
    captured: dict[str, object] = {}

    class _FakeStrategy:
        name = "fake"
        supported_feature_levels = frozenset({"patch"})
        output_representation_kind = "single_vector"
        supported_representation_kinds = frozenset({"single_vector"})
        search_database: list[object] = []

        def __init__(self, params=None, **kwargs) -> None:
            self._params = dict(params or {})
            self._kwargs = kwargs

        def hyperparam_values(self) -> dict[str, object]:
            return dict(self._params)

        def build_database(self, database_representations) -> None:
            self.search_database = []

        def search(self, query_representation, filter_same_patient=True):
            _ = query_representation, filter_same_patient
            return SimpleNamespace(
                query_id="query",
                hits=[],
                metadata=RetrievalItemMetadata(),
            )

    def _build_representation_strategy(name, *args, **kwargs):
        captured["retrieval_representation_name"] = name
        captured["retrieval_representation_params"] = kwargs.get("params")
        return _FakeStrategy(**kwargs)

    def _build_search_strategy(name, *args, **kwargs):
        captured["search_strategy_name"] = name
        captured["search_strategy_params"] = kwargs.get("params")
        return _FakeStrategy(**kwargs)

    monkeypatch.setattr(
        slide_retrieval_task_module,
        "import_representation_strategy_modules",
        lambda: None,
    )
    monkeypatch.setattr(
        slide_retrieval_task_module,
        "import_search_strategy_modules",
        lambda: None,
    )
    monkeypatch.setattr(
        slide_retrieval_task_module,
        "build_representation_strategy",
        _build_representation_strategy,
    )
    monkeypatch.setattr(
        slide_retrieval_task_module,
        "build_search_strategy",
        _build_search_strategy,
    )
    monkeypatch.setattr(
        SlideRetrievalTask,
        "_ensure_representations",
        lambda self, datasets_by_use, combo_cfg: {
            "reference": [
                RetrievalRepresentation(
                    sample_id="ref",
                    representation_type="single_vector",
                    data=[1.0],
                    metadata=RetrievalItemMetadata(patient_id="p1"),
                )
            ],
            "query": [
                RetrievalRepresentation(
                    sample_id="query",
                    representation_type="single_vector",
                    data=[1.0],
                    metadata=RetrievalItemMetadata(patient_id="p2"),
                )
            ],
        },
    )
    monkeypatch.setattr(
        SlideRetrievalTask,
        "_write_outputs",
        lambda self, results, metrics: tmp_path / "out",
    )

    combo_cfg = ComboConfig(
        feature_extraction="uni",
        feature_extraction_params={},
        tile_px=256,
        tile_px_params={},
        tile_mpp=0.5,
        tile_mpp_params={},
        retrieval_representation="splice_rgb",
        retrieval_representation_params={"percentile_threshold": 25},
        search_strategy="yottixel",
        search_strategy_params={"k": 10},
    )
    datasets_by_use = {
        "reference": [_FakeBagDataset(tiling_id="bag_a", aggregation_level="slide")],
        "query": [_FakeBagDataset(tiling_id="bag_a", aggregation_level="slide")],
    }

    result = task.execute(combo_cfg=combo_cfg, datasets_by_use=datasets_by_use)

    assert result["num_queries"] == 1
    assert captured["retrieval_representation_name"] == "splice_rgb"
    assert captured["retrieval_representation_params"] == {
        "percentile_threshold": 25
    }
    assert captured["search_strategy_name"] == "yottixel"
    assert captured["search_strategy_params"] == {"k": 10}
