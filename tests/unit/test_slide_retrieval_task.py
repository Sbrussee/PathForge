from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import pathforge.core.tasks.slide_retrieval as slide_retrieval_task_module
from pathforge.core.tasks.slide_retrieval import SlideRetrievalTask
from pathforge.core.experiments.combinations import ComboConfig
from pathforge.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)


class _FakeSlideRetrievalBagDataset:
    def __init__(self, *, tiling_id: str, aggregation_level: str, num_bags: int = 0) -> None:
        self.tiling_id = tiling_id
        self.aggregation_level = aggregation_level
        self.num_bags = num_bags
        self.sample_loader = None
        self.name = "fake_ds"

    def bind_sample_loader(self, sample_loader) -> None:
        self.sample_loader = sample_loader

    def clear_sample_loader(self) -> None:
        self.sample_loader = None

    def get_feature_level(self) -> str:
        return "patch"

    def get_feature_level_reason(self) -> str:
        return ""


def _make_task(
    tmp_path: Path,
    *,
    search_workers: int | None = None,
) -> SlideRetrievalTask:
    slide_retrieval_kwargs = {"exclusion_level": "patient"}
    if search_workers is not None:
        slide_retrieval_kwargs["search_workers"] = search_workers
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(aggregation_level="slide"),
        slide_retrieval=SimpleNamespace(**slide_retrieval_kwargs),
    )
    experiment = SimpleNamespace(cfg=cfg, project_root=str(tmp_path))
    return SlideRetrievalTask(experiment)


def test_validate_dataset_context_accepts_shared_context(tmp_path: Path) -> None:
    task = _make_task(tmp_path)

    task._validate_dataset_context(
        datasets_by_use={
            "reference": [
                _FakeSlideRetrievalBagDataset(
                    tiling_id="256px_0.5mpp",
                    aggregation_level="slide",
                )
            ],
            "query": [
                _FakeSlideRetrievalBagDataset(
                    tiling_id="256px_0.5mpp",
                    aggregation_level="slide",
                )
            ],
        },
        tiling_id="256px_0.5mpp",
        aggregation_level="slide",
    )


def test_validate_dataset_context_rejects_mixed_tiling_ids(tmp_path: Path) -> None:
    task = _make_task(tmp_path)

    with pytest.raises(ValueError, match="expected tiling_id"):
        task._validate_dataset_context(
            datasets_by_use={
                "reference": [
                    _FakeSlideRetrievalBagDataset(
                        tiling_id="256px_0.5mpp",
                        aggregation_level="slide",
                    )
                ],
                "query": [
                    _FakeSlideRetrievalBagDataset(
                        tiling_id="512px_0.5mpp",
                        aggregation_level="slide",
                    )
                ],
            },
            tiling_id="256px_0.5mpp",
            aggregation_level="slide",
        )


def test_split_representations_by_use_supports_query_reference(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    reference_only = RetrievalRepresentation(sample_id="ref", data=[1.0])
    shared_item = RetrievalRepresentation(sample_id="shared", data=[2.0])
    query_only = RetrievalRepresentation(sample_id="query", data=[3.0])

    reference_items, query_items = task._split_representations_by_use(
        representations_by_use={
            "reference": [reference_only],
            "query_reference": [shared_item],
            "query": [query_only],
        }
    )

    assert reference_items == [reference_only, shared_item]
    assert query_items == [query_only, shared_item]


def test_split_representations_by_use_can_exclude_query_reference_from_queries(
    tmp_path: Path,
) -> None:
    task = _make_task(tmp_path)
    reference_only = RetrievalRepresentation(sample_id="ref", data=[1.0])
    shared_item = RetrievalRepresentation(sample_id="shared", data=[2.0])
    query_only = RetrievalRepresentation(sample_id="query", data=[3.0])

    reference_items, query_items = task._split_representations_by_use(
        representations_by_use={
            "reference": [reference_only],
            "query_reference": [shared_item],
            "query": [query_only],
        },
        include_query_reference_as_queries=False,
    )

    assert reference_items == [reference_only, shared_item]
    assert query_items == [query_only]


def test_resolve_search_workers_caps_to_query_count(tmp_path: Path) -> None:
    task = _make_task(tmp_path, search_workers=8)

    assert task._resolve_search_workers(num_queries=3) == 3


def test_run_search_items_preserves_order_with_threads(tmp_path: Path) -> None:
    task = _make_task(tmp_path)

    results = task._run_search_items(
        search_items=["slow", "fast", "middle"],
        search_fn=lambda item: f"result-{item}",
        search_workers=3,
    )

    assert results == ["result-slow", "result-fast", "result-middle"]


def test_validate_combination_compatibility_accepts_supported_registered_pair(
    tmp_path: Path,
) -> None:
    task = _make_task(tmp_path)

    is_valid, reason = task._validate_combination_compatibility(
        datasets_by_use={
            "reference": [
                _FakeSlideRetrievalBagDataset(
                    tiling_id="256px_0.5mpp",
                    aggregation_level="slide",
                )
            ]
        },
        representation_name="splice-features",
        search_strategy_name="sish",
        aggregation_level="slide",
        exclusion_level="patient",
    )

    assert is_valid
    assert reason == ""


def test_validate_combination_compatibility_rejects_mismatched_representation_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _make_task(tmp_path)
    monkeypatch.setattr(
        slide_retrieval_task_module,
        "get_representation_strategy_output_kind",
        lambda _: "single_vector",
    )
    monkeypatch.setattr(
        slide_retrieval_task_module,
        "get_search_strategy_supported_representation_kinds",
        lambda _: frozenset({"patch_vector"}),
    )

    is_valid, reason = task._validate_combination_compatibility(
        datasets_by_use={
            "reference": [
                _FakeSlideRetrievalBagDataset(
                    tiling_id="256px_0.5mpp",
                    aggregation_level="slide",
                )
            ]
        },
        representation_name="splice-features",
        search_strategy_name="sish",
        aggregation_level="slide",
        exclusion_level="patient",
    )

    assert not is_valid
    assert "single_vector" in reason


def test_execute_requires_retrieval_dataset_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _make_task(tmp_path)
    combo_cfg = ComboConfig(
        tile_px=256,
        tile_px_params={},
        tile_mpp=0.5,
        tile_mpp_params={},
        feature_extraction="uni",
        feature_extraction_params={},
        retrieval_representation="splice-features",
        retrieval_representation_params={},
        search_strategy="sish",
        search_strategy_params={},
    )
    monkeypatch.setattr(
        "pathforge.core.tasks.slide_retrieval.build_representation_strategy",
        lambda *_, **__: SimpleNamespace(
            hyperparam_values=lambda: {},
            output_representation_kind="patch_vector",
            name="dummy_representation",
            supported_feature_levels=frozenset({"patch"}),
            load_sample=lambda **__: {},
        ),
    )

    with pytest.raises(TypeError, match="requires SlideRetrievalBagDataset"):
        task.execute(
            combo_cfg=combo_cfg,
            datasets_by_use={
                "reference": [
                    _FakeSlideRetrievalBagDataset(
                        tiling_id="256px_0.5mpp",
                        aggregation_level="slide",
                        num_bags=1,
                    )
                ],
                "query": [],
            },
        )


def test_execute_raises_when_representation_creation_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _make_task(tmp_path)
    combo_cfg = ComboConfig(
        tile_px=256,
        tile_px_params={},
        tile_mpp=0.5,
        tile_mpp_params={},
        feature_extraction="uni",
        feature_extraction_params={},
        retrieval_representation="splice-features",
        retrieval_representation_params={},
        search_strategy="sish",
        search_strategy_params={},
    )
    monkeypatch.setattr(
        "pathforge.core.tasks.slide_retrieval.build_representation_strategy",
        lambda *_, **__: SimpleNamespace(
            hyperparam_values=lambda: {},
            output_representation_kind="patch_vector",
            name="dummy_representation",
            supported_feature_levels=frozenset({"patch"}),
            load_sample=lambda **__: {},
        ),
    )

    task._validate_dataset_context = lambda **_: None  # type: ignore[method-assign]
    task._validate_combination_compatibility = lambda **_: (True, "")  # type: ignore[method-assign]
    monkeypatch.setattr(
        "pathforge.core.tasks.slide_retrieval.SlideRetrievalBagDataset",
        _FakeSlideRetrievalBagDataset,
    )
    task._collect_existing_representations = lambda **_: (  # type: ignore[method-assign]
        [RetrievalRepresentation(sample_id="ref", data=[1.0])],
        [],
    )
    task.compute_retrieval_representations = lambda **_: (  # type: ignore[method-assign]
        [RetrievalRepresentation(sample_id="qry", data=[2.0])],
        {"slide-a": "traceback text"},
    )

    with pytest.raises(RuntimeError, match="slide-a"):
        task.execute(
            combo_cfg=combo_cfg,
            datasets_by_use={
                "reference": [
                    _FakeSlideRetrievalBagDataset(
                        tiling_id="256px_0.5mpp",
                        aggregation_level="slide",
                    )
                ],
                "query": [],
            },
        )
