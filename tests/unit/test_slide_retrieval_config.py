from __future__ import annotations

from pathlib import Path

import pytest

from pathforge.config.config import Config


def test_slide_retrieval_config_does_not_require_mil(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text("dataset,slide,category\ntrain,S1,tumor\n", encoding="utf-8")

    cfg = Config.model_validate(
        {
            "experiment": {
                "project_name": "retrieval_project",
                "annotation_file": str(annotation_path),
                "task": "slide_retrieval",
                "mode": "benchmark",
            },
            "slide_retrieval": {
                "search_workers": 4,
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

    assert cfg.experiment.task == "slide_retrieval"
    assert cfg.slide_retrieval is not None
    assert cfg.slide_retrieval.search_workers == 4


def test_slide_retrieval_config_rejects_mil_dataset_use_semantics(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text("dataset,slide,category\ntrain,S1,tumor\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid dataset used_for values"):
        Config.model_validate(
            {
                "experiment": {
                    "project_name": "retrieval_project",
                    "annotation_file": str(annotation_path),
                    "task": "slide_retrieval",
                    "mode": "benchmark",
                },
                "datasets": [
                    {
                        "name": "train",
                        "slides_dir": str(tmp_path),
                        "artifacts_dir": str(tmp_path),
                        "used_for": "training",
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


def test_slide_retrieval_config_accepts_strategy_hyperparams(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text("dataset,slide,category\ntrain,S1,tumor\n", encoding="utf-8")

    cfg = Config.model_validate(
        {
            "experiment": {
                "project_name": "retrieval_project",
                "annotation_file": str(annotation_path),
                "task": "slide_retrieval",
                "mode": "benchmark",
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
                "retrieval_representation": [
                    "sdm-features",
                    {"splice-rgb": {"percentile_threshold": 25}},
                ],
                "search_strategy": [
                    {"yottixel": {"k": 10}},
                ],
                "mil": [],
            },
        }
    )

    assert cfg.benchmark_parameters.get_entries("retrieval_representation")[1].hyperparams == {
        "percentile_threshold": 25
    }
    assert cfg.benchmark_parameters.get_entries("search_strategy")[0].hyperparams == {
        "k": 10
    }


def test_slide_retrieval_config_rejects_unknown_strategy_hyperparam(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text("dataset,slide,category\ntrain,S1,tumor\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown hyperparams"):
        Config.model_validate(
            {
                "experiment": {
                    "project_name": "retrieval_project",
                    "annotation_file": str(annotation_path),
                    "task": "slide_retrieval",
                    "mode": "benchmark",
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
                    "retrieval_representation": [
                        {"splice-rgb": {"does_not_exist": 25}},
                    ],
                    "mil": [],
                },
            }
        )
