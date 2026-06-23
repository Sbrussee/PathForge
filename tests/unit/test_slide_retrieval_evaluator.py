from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathbench.config.config import Config
from pathbench.core.evaluation.slide_retrieval.adapter import (
    SlideRetrievalEvaluationAdapter,
)
from pathbench.core.experiments.base import Experiment
from pathbench.core.experiments.combinations import ComboConfig
from pathbench.slide_retrieval.io import (
    build_slide_retrieval_output_root,
    write_slide_retrieval_results_csv,
)
from pathbench.slide_retrieval.search_strategies.types import SearchHit, SearchResult


def _make_cfg(
    *,
    tmp_path: Path,
    annotation_path: Path,
    aggregation_level: str = "slide",
) -> Config:
    return Config.model_validate(
        {
            "experiment": {
                "project_name": "retrieval_eval_project",
                "annotation_file": str(annotation_path),
                "project_root": str((tmp_path / "projects").resolve()),
                "task": "slide_retrieval",
                "mode": "benchmark",
                "aggregation_level": aggregation_level,
            },
            "evaluation": {
                "label_column": "category",
                "metrics": ["hit_at_5"],
            },
            "slide_retrieval": {
                "exclusion_level": "patient",
            },
            "datasets": [
                {
                    "name": "reference_ds",
                    "slides_dir": str(tmp_path),
                    "artifacts_dir": str(tmp_path),
                    "used_for": "reference",
                },
                {
                    "name": "query_ds",
                    "slides_dir": str(tmp_path),
                    "artifacts_dir": str(tmp_path),
                    "used_for": "query",
                },
            ],
            "benchmark_parameters": {
                "tile_px": [256],
                "tile_mpp": [0.5],
                "feature_extraction": ["resnet18"],
                "retrieval_representation": ["sdm-features"],
                "search_strategy": ["yottixel"],
                "mil": [],
            },
        }
    )


def _write_run_files(
    *,
    experiment: Experiment,
    combo_cfg: ComboConfig,
    manifest: dict[str, object] | None = None,
    results: list[SearchResult] | None = None,
) -> Path:
    output_root = build_slide_retrieval_output_root(
        project_root=str(experiment.project_root),
        tiling_id="256px_0.5mpp",
        feature_name="resnet18",
        slide_representation=str(combo_cfg.get("retrieval_representation")),
        search_method=str(combo_cfg.get("search_strategy")),
    )
    run_dir = output_root / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            manifest
            or {
                "tiling_id": "256px_0.5mpp",
                "aggregation_level": str(experiment.cfg.experiment.aggregation_level),
                "feature_extraction": "resnet18",
                "slide_representation": "sdm-features",
                "search_method": "yottixel",
            }
        ),
        encoding="utf-8",
    )
    write_slide_retrieval_results_csv(
        run_dir / "query_results.csv",
        results
        or [
            SearchResult(
                query_sample_id="S1",
                hits=[
                    SearchHit(sample_id="S2", score=0.9, rank=1),
                    SearchHit(sample_id="S3", score=0.8, rank=2),
                ],
            )
        ],
    )
    return run_dir


def test_slide_retrieval_evaluator_discovers_runs_and_loads_labels(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text(
        "dataset,slide,patient,category\n"
        "query_ds,S1,P1,tumor\n"
        "reference_ds,S2,P2,normal\n"
        "reference_ds,S3,P3,tumor\n",
        encoding="utf-8",
    )
    cfg = _make_cfg(tmp_path=tmp_path, annotation_path=annotation_path)
    experiment = Experiment(cfg)
    combo_cfg = ComboConfig(
        tile_px=256,
        tile_px_params={},
        tile_mpp=0.5,
        tile_mpp_params={},
        feature_extraction="resnet18",
        feature_extraction_params={},
        retrieval_representation="sdm-features",
        retrieval_representation_params={},
        search_strategy="yottixel",
        search_strategy_params={},
    )
    _write_run_files(experiment=experiment, combo_cfg=combo_cfg)

    evaluation_adapter = SlideRetrievalEvaluationAdapter(experiment)
    run_contexts = evaluation_adapter.discover_runs()

    assert len(run_contexts) == 1
    data = evaluation_adapter.load_run_data(run_contexts[0])
    assert data.queries[0].query_id == "S1"
    assert data.queries[0].query_label == "tumor"
    assert [hit.label for hit in data.queries[0].hits] == ["normal", "tumor"]


def test_slide_retrieval_evaluator_reports_all_missing_labels(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text(
        "dataset,slide,patient,category\n"
        "query_ds,S1,P1,tumor\n",
        encoding="utf-8",
    )
    cfg = _make_cfg(tmp_path=tmp_path, annotation_path=annotation_path)
    experiment = Experiment(cfg)
    combo_cfg = ComboConfig(
        tile_px=256,
        tile_px_params={},
        tile_mpp=0.5,
        tile_mpp_params={},
        feature_extraction="resnet18",
        feature_extraction_params={},
        retrieval_representation="sdm-features",
        retrieval_representation_params={},
        search_strategy="yottixel",
        search_strategy_params={},
    )
    _write_run_files(
        experiment=experiment,
        combo_cfg=combo_cfg,
        results=[
            SearchResult(
                query_sample_id="S2",
                hits=[SearchHit(sample_id="S3", score=0.9, rank=1)],
            )
        ],
    )

    evaluation_adapter = SlideRetrievalEvaluationAdapter(experiment)
    run_context = evaluation_adapter.discover_runs()[0]

    with pytest.raises(ValueError, match="S2") as error:
        evaluation_adapter.load_run_data(run_context)

    assert "S3" in str(error.value)


def test_slide_retrieval_evaluator_reports_inconsistent_aggregated_labels(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text(
        "dataset,slide,patient,category\n"
        "query_ds,S1,P1,tumor\n"
        "query_ds,S2,P1,normal\n"
        "reference_ds,S3,P2,tumor\n",
        encoding="utf-8",
    )
    cfg = _make_cfg(
        tmp_path=tmp_path,
        annotation_path=annotation_path,
        aggregation_level="patient",
    )
    experiment = Experiment(cfg)
    combo_cfg = ComboConfig(
        tile_px=256,
        tile_px_params={},
        tile_mpp=0.5,
        tile_mpp_params={},
        feature_extraction="resnet18",
        feature_extraction_params={},
        retrieval_representation="sdm-features",
        retrieval_representation_params={},
        search_strategy="yottixel",
        search_strategy_params={},
    )
    _write_run_files(
        experiment=experiment,
        combo_cfg=combo_cfg,
        results=[
            SearchResult(
                query_sample_id="P1",
                hits=[SearchHit(sample_id="P2", score=0.9, rank=1)],
            )
        ],
    )

    evaluation_adapter = SlideRetrievalEvaluationAdapter(experiment)
    run_context = evaluation_adapter.discover_runs()[0]

    with pytest.raises(ValueError, match="Inconsistent aggregated labels") as error:
        evaluation_adapter.load_run_data(run_context)

    assert "P1" in str(error.value)
