from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import pathbench.cli.evaluate as evaluate_cli
from pathbench.config.config import Config
from pathbench.core.experiments.base import Experiment
from pathbench.slide_retrieval.io import (
    build_slide_retrieval_output_root,
    write_slide_retrieval_results_csv,
)
from pathbench.slide_retrieval.search_strategies.types import SearchHit, SearchResult


@pytest.mark.smoke
def test_smoke_evaluate_cli_writes_metrics_json(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text(
        "dataset,slide,patient,category\n"
        "query_ds,S1,P1,tumor\n"
        "reference_ds,S2,P2,tumor\n",
        encoding="utf-8",
    )
    project_root_base = (tmp_path / "projects").resolve()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "project_name": "eval_smoke_project",
                    "annotation_file": str(annotation_path),
                    "project_root": str(project_root_base),
                    "task": "slide_retrieval",
                    "mode": "benchmark",
                    "aggregation_level": "slide",
                },
                "evaluation": {
                    "label_column": "category",
                    "metrics": ["hit_at_1"],
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
                    "retrieval_representation": ["sdm_features"],
                    "search_strategy": ["yottixel"],
                    "mil": [],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    cfg = Config.from_yaml(config_path)
    experiment = Experiment(cfg)
    output_root = build_slide_retrieval_output_root(
        project_root=str(experiment.project_root),
        tiling_id="256px_0.5mpp",
        feature_name="resnet18",
        slide_representation="sdm_features",
        search_method="yottixel",
    )
    run_dir = output_root / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "tiling_id": "256px_0.5mpp",
                "aggregation_level": "slide",
                "feature_extraction": "resnet18",
                "slide_representation": "sdm_features",
                "search_method": "yottixel",
            }
        ),
        encoding="utf-8",
    )
    write_slide_retrieval_results_csv(
        run_dir / "query_results.csv",
        [
            SearchResult(
                query_sample_id="S1",
                hits=[SearchHit(sample_id="S2", score=0.95, rank=1)],
            )
        ],
    )

    exit_code = evaluate_cli.main(["--config", str(config_path)])

    metrics_path = run_dir / "evaluation" / "metrics.json"
    assert exit_code == 0
    assert metrics_path.is_file()

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["hit_at_1"]["micro"] == pytest.approx(1.0)
