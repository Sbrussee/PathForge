"""Exercise precomputation and retrieval against the same physical-slide caches."""

import time
import tracemalloc
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import pathforge.cli.retrieval_representations as cli
import pathforge.core.tasks.slide_retrieval as task_module
from pathforge.core.experiments.combinations import ComboConfig
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.core.io.slide_artifacts.features import write_features
from pathforge.core.io.slide_artifacts.tiles import write_coords
from pathforge.core.tasks.slide_retrieval import SlideRetrievalTask
from pathforge.slide_retrieval import representation_workflow as workflow
from pathforge.slide_retrieval.representation_strategies.types import (
    RetrievalRepresentation,
)
from pathforge.slide_retrieval.search_strategies.types import SearchResult
from tests.unit.test_representation_workflow import Dataset, sample, save_cached


@pytest.mark.smoke
def test_precompute_then_retrieve_reuses_physical_caches(
    tmp_path, monkeypatch, record_property
):
    dataset = Dataset(tmp_path, [sample(tmp_path, "patient", ["b", "a"])])
    dataset.aggregation_level = "patient"
    for path in dataset.get_sample(0).artifact_paths:
        with FileHandleH5(path, mode="a") as artifact:
            write_features(artifact, "tiles", "features", np.ones((2, 2)))
            write_coords(artifact, "tiles", np.zeros((2, 5), dtype=np.int32))
    save_cached(tmp_path, "a", [[1.0, 2.0]])

    cfg = SimpleNamespace(
        experiment=SimpleNamespace(aggregation_level="patient"),
        slide_retrieval=SimpleNamespace(exclusion_level="patient"),
    )
    task = SlideRetrievalTask(SimpleNamespace(cfg=cfg, project_root=str(tmp_path)))
    combo = ComboConfig(
        tile_px=256, tile_px_params={}, tile_mpp=0.5, tile_mpp_params={},
        feature_extraction="uni", feature_extraction_params={},
        retrieval_representation="splice-features", retrieval_representation_params={},
        search_strategy="yottixel", search_strategy_params={},
    )
    calls = []

    class Strategy:
        def load_sample(self, **_):
            return {}

        def run(self, *, sample, **_):
            calls.append(sample.sample_id)
            return RetrievalRepresentation(
                sample_id=sample.sample_id, data=np.array([[3.0, 4.0]])
            )

    class Search:
        def build_database(self, representations):
            self.search_database = representations

        def search(self, representation):
            np.testing.assert_array_equal(representation.data, [[3, 4], [1, 2]])
            return SearchResult(query_sample_id=representation.sample_id, hits=[])

        def hyperparam_values(self):
            return {}

    for module in [cli, task_module]:
        monkeypatch.setattr(module, "build_tiling_id", lambda _: "tiles")
        monkeypatch.setattr(module, "build_feature_name", lambda _: "features")
        monkeypatch.setattr(module, "build_retrieval_representation_id", lambda **_: "rep")
    monkeypatch.setattr(cli, "build_representation_strategy", lambda *_, **__: Strategy())

    started = time.perf_counter()
    tracemalloc.start()
    try:
        output = cli._materialize_representations_for_combo(
            task=task, combo_cfg=combo, datasets_by_use={"query_reference": [dataset]}
        )
        assert output["num_cached"] == output["num_created"] == 1
        assert calls == ["b"]

        monkeypatch.setattr(task_module, "inspect_retrieval_inputs", lambda *_, **__: pytest.fail("cached retrieval inspected original features"))
        monkeypatch.setattr(task_module, "build_representation_strategy", lambda *_, **__: pytest.fail("cached retrieval constructed a selector"))
        monkeypatch.setattr(task_module, "build_search_strategy", lambda *_, **__: Search())
        loader = Mock(wraps=workflow.load_slide_retrieval_representation)
        monkeypatch.setattr(workflow, "load_slide_retrieval_representation", loader)
        result = task.execute(combo_cfg=combo, datasets_by_use={"query_reference": [dataset]})
        assert result["num_queries"] == result["num_reference_items"] == 1
        assert loader.call_count == 2
        assert (tmp_path / result["output_dir"] / "manifest.json").is_file()
        assert (tmp_path / result["output_dir"] / "query_results.xlsx").is_file()
    finally:
        record_property("elapsed_seconds", time.perf_counter() - started)
        record_property("peak_memory_bytes", tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
