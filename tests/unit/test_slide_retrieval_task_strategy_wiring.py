from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import pathbench.benchmarking.tasks.slide_retrieval as slide_retrieval_task_module
from pathbench.benchmarking.tasks.slide_retrieval import SlideRetrievalTask
from pathbench.core.experiments.combinations import ComboConfig
from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathbench.slide_retrieval.search_strategies.types import SearchResult


class _FakeBagDataset:
    def __init__(
        self,
        *,
        tiling_id: str,
        aggregation_level: str,
        sample_id: str,
    ) -> None:
        self.tiling_id = tiling_id
        self.aggregation_level = aggregation_level
        self.sample_id = sample_id
        self.sample_loader = None
        self.name = sample_id

    def get_feature_level(self) -> str:
        return "patch"

    def bind_sample_loader(self, sample_loader) -> None:
        self.sample_loader = sample_loader

    def clear_sample_loader(self) -> None:
        self.sample_loader = None


class _FakeRepresentationStrategy:
    name = "fake_representation"
    supported_feature_levels = frozenset({"patch"})
    output_representation_kind = "patch_vector"

    def __init__(self, params=None, **kwargs) -> None:
        self._params = dict(params or {})
        self._kwargs = dict(kwargs)

    def hyperparam_values(self) -> dict[str, object]:
        return dict(self._params)


class _FakeSearchStrategy:
    name = "fake_search"
    supported_representation_kinds = frozenset({"patch_vector"})

    def __init__(self, params=None, **kwargs) -> None:
        self._params = dict(params or {})
        self._kwargs = dict(kwargs)
        self.search_database: list[object] = []

    def hyperparam_values(self) -> dict[str, object]:
        return dict(self._params)

    def build_database(self, database_representations) -> None:
        self.search_database = list(database_representations)

    def search(self, query_representation, **kwargs) -> SearchResult:
        _ = kwargs
        return SearchResult(query_sample_id=query_representation.sample_id, hits=[])


def _make_task(tmp_path: Path) -> SlideRetrievalTask:
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(aggregation_level="slide"),
        slide_retrieval=SimpleNamespace(exclusion_level="patient"),
    )
    experiment = SimpleNamespace(cfg=cfg, project_root=str(tmp_path))
    return SlideRetrievalTask(experiment)


def test_execute_passes_only_generic_context_to_strategy_builders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _make_task(tmp_path)
    captured: dict[str, dict[str, object]] = {}

    def _build_representation_strategy(name, *args, **kwargs):
        captured["representation"] = {
            "name": name,
            "kwargs": dict(kwargs),
        }
        return _FakeRepresentationStrategy(*args, **kwargs)

    def _build_search_strategy(name, *args, **kwargs):
        captured["search"] = {
            "name": name,
            "kwargs": dict(kwargs),
        }
        return _FakeSearchStrategy(*args, **kwargs)

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
        slide_retrieval_task_module,
        "SlideRetrievalBagDataset",
        _FakeBagDataset,
    )
    monkeypatch.setattr(
        SlideRetrievalTask,
        "_collect_existing_representations",
        lambda self, **kwargs: (
            [
                RetrievalRepresentation(
                    sample_id=kwargs["bag_dataset"].sample_id,
                    data=[[3.0, 4.0]],
                )
            ],
            None,
        ),
    )

    combo_cfg = ComboConfig(
        tile_px=256,
        tile_px_params={},
        tile_mpp=0.5,
        tile_mpp_params={},
        feature_extraction="uni",
        feature_extraction_params={},
        retrieval_representation="splice-features",
        retrieval_representation_params={"percentile_threshold": 25},
        search_strategy="sish",
        search_strategy_params={"k": 5},
    )
    datasets_by_use = {
        "reference": [
            _FakeBagDataset(
                tiling_id="256px_0.5mpp",
                aggregation_level="slide",
                sample_id="ref-1",
            )
        ],
        "query": [
            _FakeBagDataset(
                tiling_id="256px_0.5mpp",
                aggregation_level="slide",
                sample_id="query-1",
            )
        ],
    }

    result = task.execute(combo_cfg=combo_cfg, datasets_by_use=datasets_by_use)

    assert result["num_queries"] == 1
    assert captured["representation"]["name"] == "splice-features"
    assert captured["search"]["name"] == "sish"
    assert captured["search"]["kwargs"]["config"] is task.cfg
    assert "project_root" not in captured["representation"]["kwargs"]
    assert "project_root" not in captured["search"]["kwargs"]
