from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pathbench.core.tasks.slide_retrieval import SlideRetrievalTask
from pathbench.core.experiments.combinations import ComboConfig
from pathbench.slide_retrieval.representation_strategies.types import RetrievalRepresentation
from ._smoke_dataset import attach_smoke_outputs, capture_smoke_metrics
from .conftest import RetrievalDatasets


def _make_task(tmp_path: Path) -> SlideRetrievalTask:
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(aggregation_level="slide", num_workers=0),
        slide_retrieval=SimpleNamespace(exclusion_level="patient"),
    )
    return SlideRetrievalTask(SimpleNamespace(cfg=cfg, project_root=str(tmp_path)))


def _register_retrieval_strategies() -> None:
    from pathbench.slide_retrieval.representation_strategies.registry import (
        import_representation_strategy_modules,
    )
    from pathbench.slide_retrieval.search_strategies.registry import (
        import_search_strategy_modules,
    )

    import_representation_strategy_modules()
    import_search_strategy_modules()


@pytest.mark.smoke
def test_smoke_slide_retrieval_benchmark_writes_manifest_and_ranked_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retrieval_wsi_datasets: RetrievalDatasets,
    slide_level_feature_matrix: tuple,
) -> None:
    """Slide retrieval benchmark must write a valid manifest.json and query_results.csv."""
    _register_retrieval_strategies()
    slide_ids, feature_matrix = slide_level_feature_matrix

    def _real_feature_cache(
        self,
        *,
        bag_dataset,
        representation_id,
        aggregation_level,
        exclusion_level,
    ):
        reps = []
        for i in range(bag_dataset.num_bags):
            sample = bag_dataset.get_sample(i)
            try:
                idx = slide_ids.index(sample.sample_id)
                data = [feature_matrix[idx].tolist()]
            except ValueError:
                data = [feature_matrix[0].tolist()]
            reps.append(
                RetrievalRepresentation(
                    sample_id=sample.sample_id,
                    data=data,
                    exclusion_key=sample.patient_id,
                )
            )
        return reps, None

    monkeypatch.setattr(
        SlideRetrievalTask, "_collect_existing_representations", _real_feature_cache
    )

    task = _make_task(tmp_path)
    ref_dataset = retrieval_wsi_datasets.reference
    qry_dataset = retrieval_wsi_datasets.query

    combo_cfg = ComboConfig(
        tile_px=224,
        tile_px_params={},
        tile_mpp=1.0,
        tile_mpp_params={},
        feature_extraction="resnet18",
        feature_extraction_params={},
        retrieval_representation="hshr_features",
        retrieval_representation_params={},
        search_strategy="yottixel",
        search_strategy_params={"k": 5},
    )
    datasets_by_use = {
        "reference": [ref_dataset],
        "query": [qry_dataset],
    }

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="smoke_slide_retrieval_benchmark",
        metadata={
            "num_queries": qry_dataset.num_bags,
            "num_reference": ref_dataset.num_bags,
            "representation": "hshr_features",
            "search_strategy": "yottixel",
        },
    ) as metadata:
        result = task.execute(combo_cfg=combo_cfg, datasets_by_use=datasets_by_use)

        output_dir = Path(result["output_dir"])
        manifest_path = output_dir / "manifest.json"
        results_path = output_dir / "query_results.csv"

        attach_smoke_outputs(
            metadata,
            step_name="smoke_slide_retrieval_benchmark",
            final={
                "manifest_json": manifest_path,
                "query_results_csv": results_path,
            },
        )

    assert manifest_path.is_file()
    assert results_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["slide_representation"] == "hshr_features"
    assert manifest["search_method"] == "yottixel"
    assert manifest["num_queries"] == qry_dataset.num_bags
    assert manifest["num_reference_items"] == ref_dataset.num_bags

    with results_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames is not None
    assert "query_sample_id" in reader.fieldnames
    assert len(rows) == qry_dataset.num_bags
    qry_ids = {qry_dataset.get_sample(i).sample_id for i in range(qry_dataset.num_bags)}
    assert all(row["query_sample_id"] in qry_ids for row in rows)
