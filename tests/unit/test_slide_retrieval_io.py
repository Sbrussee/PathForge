from __future__ import annotations

import csv
from pathlib import Path

from pathforge.slide_retrieval.io import (
    build_slide_retrieval_representation_root,
    build_slide_retrieval_output_root,
    write_metrics_csv,
)


def test_write_metrics_csv_writes_flat_metric_rows(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    metrics = {
        "hit_at_5": {
            "k": 5,
            "per_class": {
                "tumor": 1.0,
                "normal": 0.5,
            },
            "macro": 0.75,
            "micro": 0.8,
            "num_queries": 10,
            "insufficient_k_queries": 2,
        }
    }

    write_metrics_csv(metrics_path, metrics)

    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {
            "metric": "hit_at_5",
            "scope": "per_class",
            "label": "tumor",
            "value": "1.0",
        },
        {
            "metric": "hit_at_5",
            "scope": "per_class",
            "label": "normal",
            "value": "0.5",
        },
        {
            "metric": "hit_at_5",
            "scope": "macro",
            "label": "",
            "value": "0.75",
        },
        {
            "metric": "hit_at_5",
            "scope": "micro",
            "label": "",
            "value": "0.8",
        },
        {
            "metric": "hit_at_5",
            "scope": "k",
            "label": "",
            "value": "5",
        },
        {
            "metric": "hit_at_5",
            "scope": "num_queries",
            "label": "",
            "value": "10",
        },
        {
            "metric": "hit_at_5",
            "scope": "insufficient_k_queries",
            "label": "",
            "value": "2",
        },
    ]


def test_build_slide_retrieval_representation_root_lowercases_name_components(
    tmp_path: Path,
) -> None:
    output_root = build_slide_retrieval_representation_root(
        project_root=str(tmp_path),
        tiling_id="256px_0.5mpp",
        feature_name="UNI2_ColorNorm",
        slide_representation="Yottixel_features",
    )

    assert str(output_root).endswith(
        "eval_slide_retrieval/256px_0.5mpp_uni2_colornorm/yottixel-features"
    )


def test_build_slide_retrieval_output_root_places_search_under_representation(
    tmp_path: Path,
) -> None:
    output_root = build_slide_retrieval_output_root(
        project_root=str(tmp_path),
        tiling_id="256px_0.5mpp",
        feature_name="UNI2_ColorNorm",
        slide_representation="Yottixel_features",
        search_method="RetCCL",
    )

    assert str(output_root).endswith(
        "eval_slide_retrieval/256px_0.5mpp_uni2_colornorm/yottixel-features/retccl"
    )
