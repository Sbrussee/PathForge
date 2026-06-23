from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pathbench.core.tasks.slide_retrieval import SlideRetrievalTask
from pathbench.core.experiments.combinations import ComboConfig
from pathbench.slide_retrieval.representation_strategies.types import RetrievalRepresentation
from pathbench.slide_retrieval.search_strategies.types import SearchHit, SearchResult


class _FakeBagDataset:
    def __init__(self, *, tiling_id: str, aggregation_level: str, sample_id: str) -> None:
        self.tiling_id = tiling_id
        self.aggregation_level = aggregation_level
        self.sample_id = sample_id
        self.name = f"ds_{sample_id}"

    def get_feature_level(self) -> str:
        return "patch"


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
        self.search_database: list[RetrievalRepresentation] = []

    def hyperparam_values(self) -> dict[str, object]:
        return dict(self._params)

    def build_database(self, database_representations) -> None:
        self.search_database = list(database_representations)

    def search(self, query_representation, **kwargs) -> SearchResult:
        _ = kwargs, query_representation
        return SearchResult(
            query_sample_id="query-1",
            hits=[SearchHit(sample_id="ref-1", score=0.25, rank=1)],
        )


def _make_task(tmp_path: Path) -> SlideRetrievalTask:
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(aggregation_level="slide"),
        slide_retrieval=SimpleNamespace(exclusion_level="patient"),
    )
    experiment = SimpleNamespace(cfg=cfg, project_root=str(tmp_path))
    return SlideRetrievalTask(experiment)


@pytest.mark.smoke
def test_smoke_slide_retrieval_benchmark_writes_manifest_and_ranked_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pathbench.core.tasks.slide_retrieval as slide_retrieval_task_module

    task = _make_task(tmp_path)

    monkeypatch.setattr(
        slide_retrieval_task_module,
        "build_representation_strategy",
        lambda _name, **kwargs: _FakeRepresentationStrategy(**kwargs),
    )
    monkeypatch.setattr(
        slide_retrieval_task_module,
        "build_search_strategy",
        lambda _name, **kwargs: _FakeSearchStrategy(**kwargs),
    )
    monkeypatch.setattr(slide_retrieval_task_module, "SlideRetrievalBagDataset", _FakeBagDataset)
    monkeypatch.setattr(
        SlideRetrievalTask,
        "_collect_existing_representations",
        lambda self, **kwargs: (
            [
                RetrievalRepresentation(
                    sample_id=kwargs["bag_dataset"].sample_id,
                    data=[[1.0, 2.0]],
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
        retrieval_representation="yottixel-features",
        retrieval_representation_params={"n_clusters": 2},
        search_strategy="yottixel",
        search_strategy_params={"k": 1},
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

    output_dir = Path(result["output_dir"])
    manifest_path = output_dir / "manifest.json"
    results_path = output_dir / "query_results.xlsx"

    assert manifest_path.is_file()
    assert results_path.is_file()
    assert output_dir.parent.name == "yottixel"
    assert output_dir.parent.parent.name == "yottixel-features"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["slide_representation"] == "yottixel-features"
    assert manifest["search_method"] == "yottixel"
    assert manifest["num_queries"] == 1
    assert manifest["num_reference_items"] == 1
    assert manifest["top_k_saved"] == 1

    assert results_path.suffix == ".xlsx"
