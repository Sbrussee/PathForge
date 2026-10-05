from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.slide_retrieval.io import (
    build_slide_retrieval_representation_root,
    build_slide_retrieval_output_root,
    load_slide_retrieval_representation,
    save_slide_retrieval_representation,
    write_metrics_csv,
)
from pathforge.slide_retrieval.representation_strategies.storage import (
    build_retrieval_representation_artifact_path,
)
from pathforge.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)


def test_retrieval_artifact_path_is_one_file_per_physical_slide(
    tmp_path: Path,
) -> None:
    assert build_retrieval_representation_artifact_path(
        artifacts_dir=tmp_path,
        slide_id="HMC-T16_00166-1Ca-HE-000",
    ) == (
        tmp_path / "slide_retrieval" / "HMC-T16_00166-1Ca-HE-000.h5"
    )


def test_cached_retrieval_representation_round_trips_feature_level(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "slide_retrieval" / "slide-1.h5"
    representation = RetrievalRepresentation(
        sample_id="slide-1",
        data=np.asarray([[1.0, 2.0]], dtype=np.float32),
        feature_level="patch",
    )

    with FileHandleH5(artifact_path, mode="a") as artifact:
        save_slide_retrieval_representation(
            retrieval_artifact=artifact,
            tile_id="256px_0.5mpp",
            representation_id="uni__splice__abc123",
            entry_id=None,
            representation=representation,
        )

    with FileHandleH5(artifact_path, mode="r") as artifact:
        loaded = load_slide_retrieval_representation(
            retrieval_artifact=artifact,
            tile_id="256px_0.5mpp",
            representation_id="uni__splice__abc123",
            entry_id=None,
        )

    assert loaded is not None
    assert loaded.feature_level == "patch"


def test_save_cached_retrieval_representation_rejects_missing_feature_level(
    tmp_path: Path,
) -> None:
    with FileHandleH5(tmp_path / "slide_retrieval" / "slide-1.h5", mode="a") as artifact:
        with pytest.raises(ValueError, match="feature_level"):
            save_slide_retrieval_representation(
                retrieval_artifact=artifact,
                tile_id="256px_0.5mpp",
                representation_id="uni__splice__abc123",
                entry_id=None,
                representation=RetrievalRepresentation(
                    sample_id="slide-1",
                    data=np.asarray([[1.0, 2.0]], dtype=np.float32),
                ),
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
