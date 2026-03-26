from __future__ import annotations

from pathlib import Path

import pytest

from pathbench.config.config import Config
from pathbench.slide_retrieval.validation.metrics import compute_hit_at_k
from pathbench.slide_retrieval.validation.registry import parse_validation_metric_name
from pathbench.slide_retrieval.validation.types import (
    NormalizedSearchHit,
    NormalizedSearchResult,
)


def test_parse_validation_metric_name_parses_hit_at_k() -> None:
    request = parse_validation_metric_name("hit_at_10")

    assert request.raw_name == "hit_at_10"
    assert request.metric_name == "hit"
    assert request.registry_key == "hit_at_k"
    assert request.k == 10


def test_parse_validation_metric_name_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="Expected format"):
        parse_validation_metric_name("hit")


def test_compute_hit_at_k_returns_expected_micro_and_macro() -> None:
    results = [
        NormalizedSearchResult(
            query_id="q1",
            query_label="tumor",
            query_patient_id="p1",
            hits=[
                NormalizedSearchHit(
                    item_id="h1",
                    label="tumor",
                    patient_id="r1",
                    score=0.9,
                    rank=1,
                ),
                NormalizedSearchHit(
                    item_id="h2",
                    label="normal",
                    patient_id="r2",
                    score=0.8,
                    rank=2,
                ),
            ],
            available_k=2,
        ),
        NormalizedSearchResult(
            query_id="q2",
            query_label="normal",
            query_patient_id="p2",
            hits=[
                NormalizedSearchHit(
                    item_id="h3",
                    label="tumor",
                    patient_id="r3",
                    score=0.7,
                    rank=1,
                ),
                NormalizedSearchHit(
                    item_id="h4",
                    label="normal",
                    patient_id="r4",
                    score=0.6,
                    rank=2,
                ),
            ],
            available_k=2,
        ),
    ]

    payload = compute_hit_at_k(results, k=1)

    assert payload["hit_at_1"]["per_class"] == {"tumor": 1.0, "normal": 0.0}
    assert payload["hit_at_1"]["macro"] == pytest.approx(0.5)
    assert payload["hit_at_1"]["micro"] == pytest.approx(0.5)
    assert payload["hit_at_1"]["insufficient_k_queries"] == 0


def test_compute_hit_at_k_counts_short_results_as_miss() -> None:
    results = [
        NormalizedSearchResult(
            query_id="q1",
            query_label="tumor",
            query_patient_id="p1",
            hits=[
                NormalizedSearchHit(
                    item_id="h1",
                    label="tumor",
                    patient_id="r1",
                    score=0.9,
                    rank=1,
                )
            ],
            available_k=1,
        )
    ]

    payload = compute_hit_at_k(results, k=2)

    assert payload["hit_at_2"]["per_class"] == {"tumor": 0.0}
    assert payload["hit_at_2"]["macro"] == pytest.approx(0.0)
    assert payload["hit_at_2"]["micro"] == pytest.approx(0.0)
    assert payload["hit_at_2"]["insufficient_k_queries"] == 1


def test_slide_retrieval_config_accepts_registered_evaluation_metric(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text("dataset,slide,category\ntrain,S1,tumor\n", encoding="utf-8")

    cfg = Config.model_validate(
        {
            "experiment": {
                "project_name": "retrieval_project",
                "annotation_file": str(annotation_path),
                "task": "slide_retrieval",
                "mode": "benchmark",
                "evaluation": ["hit_at_5"],
            },
            "datasets": [
                {
                    "name": "train",
                    "slides_dir": str(tmp_path),
                    "artifacts_dir": str(tmp_path),
                    "used_for": "reference",
                }
            ],
            "benchmark_parameters": {
                "tile_px": [256],
                "tile_mpp": [0.5],
                "feature_extraction": ["resnet18"],
                "mil": [],
            },
        }
    )

    assert cfg.experiment.evaluation == ["hit_at_5"]


def test_slide_retrieval_config_rejects_unknown_evaluation_metric(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text("dataset,slide,category\ntrain,S1,tumor\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not registered"):
        Config.model_validate(
            {
                "experiment": {
                    "project_name": "retrieval_project",
                    "annotation_file": str(annotation_path),
                    "task": "slide_retrieval",
                    "mode": "benchmark",
                    "evaluation": ["map_at_5"],
                },
                "datasets": [
                    {
                        "name": "train",
                        "slides_dir": str(tmp_path),
                        "artifacts_dir": str(tmp_path),
                        "used_for": "reference",
                    }
                ],
                "benchmark_parameters": {
                    "tile_px": [256],
                    "tile_mpp": [0.5],
                    "feature_extraction": ["resnet18"],
                    "mil": [],
                },
            }
        )
