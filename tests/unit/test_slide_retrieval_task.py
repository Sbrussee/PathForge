from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pathbench.benchmarking.tasks.slide_retrieval import SlideRetrievalTask
from pathbench.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)


class _FakeSlideRetrievalBagDataset:
    def __init__(self, *, tiling_id: str, aggregation_level: str, num_bags: int = 0) -> None:
        self.tiling_id = tiling_id
        self.aggregation_level = aggregation_level
        self.num_bags = num_bags


def _make_task(tmp_path: Path) -> SlideRetrievalTask:
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(aggregation_level="slide"),
        slide_retrieval=SimpleNamespace(exclusion_level="patient"),
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


def test_load_or_compute_representations_requires_retrieval_dataset_type(
    tmp_path: Path,
) -> None:
    task = _make_task(tmp_path)
    combo_cfg = SimpleNamespace()

    with pytest.raises(TypeError, match="requires SlideRetrievalBagDataset"):
        task._load_or_compute_representations(
            datasets_by_use={
                "reference": [
                    _FakeSlideRetrievalBagDataset(
                        tiling_id="256px_0.5mpp",
                        aggregation_level="slide",
                        num_bags=1,
                    )
                ]
            },
            combo_cfg=combo_cfg,
            representation_strategy=SimpleNamespace(load_sample=lambda **_: {}, hyperparam_values=lambda: {}),
            representation_id="rep-id",
            aggregation_level="slide",
            exclusion_level="patient",
        )
