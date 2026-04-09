from __future__ import annotations

from pathlib import Path

import pytest

from pathbench.config.config import Config


def test_evaluation_config_rejects_missing_label_column_when_metrics_are_set(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text("dataset,slide,category\ntrain,S1,tumor\n", encoding="utf-8")

    with pytest.raises(ValueError, match="label_column is required"):
        Config.model_validate(
            {
                "experiment": {
                    "project_name": "retrieval_project",
                    "annotation_file": str(annotation_path),
                    "task": "slide_retrieval",
                    "mode": "benchmark",
                },
                "evaluation": {
                    "metrics": ["hit_at_5"],
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


def test_evaluation_config_rejects_unknown_label_column(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text("dataset,slide,category\ntrain,S1,tumor\n", encoding="utf-8")

    with pytest.raises(ValueError, match="was not found in annotations header"):
        Config.model_validate(
            {
                "experiment": {
                    "project_name": "retrieval_project",
                    "annotation_file": str(annotation_path),
                    "task": "slide_retrieval",
                    "mode": "benchmark",
                },
                "evaluation": {
                    "label_column": "diagnosis",
                    "metrics": ["hit_at_5"],
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
