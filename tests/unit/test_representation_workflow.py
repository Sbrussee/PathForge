from types import SimpleNamespace

import numpy as np
import pytest

from pathforge.core.datasets.bag_dataset import BagSample, SlideRetrievalBagDataset
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.slide_retrieval import representation_workflow as workflow
from pathforge.slide_retrieval.io import save_slide_retrieval_representation
from pathforge.slide_retrieval.representation_strategies.storage import build_retrieval_representation_artifact_path
from pathforge.slide_retrieval.representation_strategies.types import RetrievalRepresentation


class Dataset(SlideRetrievalBagDataset):
    def __init__(self, root, samples, name="dataset"):
        self.artifacts_dir = root
        self._samples = samples
        self._name = name
        self.tiling_id = "tiles"
        self.extractor_name = "features"

    @property
    def num_bags(self):
        return len(self._samples)

    def get_sample(self, index):
        return self._samples[index]

    def _load_slide_bag(self, path):
        return str(path)


def sample(root, identity, members, patient="patient"):
    return BagSample(
        sample_id=identity, slide_ids=members,
        artifact_paths=[root / f"{member}.h5" for member in members],
        patient_id=patient, category="category",
    )


def save_cached(root, slide_id, data):
    path = build_retrieval_representation_artifact_path(artifacts_dir=root, slide_id=slide_id)
    with FileHandleH5(path, mode="a") as artifact:
        save_slide_retrieval_representation(
            retrieval_artifact=artifact, tile_id="tiles", representation_id="rep", entry_id=None,
            representation=RetrievalRepresentation(
                sample_id=slide_id, data=np.asarray(data), feature_level="patch"
            ), params={},
        )


def test_plan_checks_presence_without_reading_arrays_and_deduplicates(tmp_path, monkeypatch):
    save_cached(tmp_path, "a", [[1, 2]])
    dataset = Dataset(tmp_path, [sample(tmp_path, "group", ["b", "a"])])
    monkeypatch.setattr(workflow, "load_slide_retrieval_representation", lambda **_: pytest.fail("planning loaded arrays"))
    plan = workflow.plan_representations(
        datasets_by_use={"reference": [dataset], "query": [dataset]}, representation_id="rep"
    )
    assert len(plan.existing) == len(plan.missing) == 1
    assert [request.sample.sample_id for request in plan.existing.values()] == ["a"]
    assert [request.sample.sample_id for request in plan.missing.values()] == ["b"]
    assert plan.groups_by_use["reference"][0].member_keys == plan.groups_by_use["query"][0].member_keys


def test_create_only_misses_then_assemble_in_group_order_without_mutation(tmp_path):
    save_cached(tmp_path, "a", [[1, 2]])
    dataset = Dataset(tmp_path, [sample(tmp_path, "group", ["b", "a"])])
    plan = workflow.plan_representations(
        datasets_by_use={"reference": [dataset], "query": [dataset]}, representation_id="rep"
    )
    created_samples = []

    class Strategy:
        def load_sample(self, *, sample, base_dataset, **_):
            assert base_dataset.load_bag(0) == str(sample.artifact_paths[0])
            return {}

        def run(self, *, sample, **_):
            created_samples.append(sample.sample_id)
            assert sample.slide_ids == ["b"]
            return RetrievalRepresentation(sample_id=sample.sample_id, data=np.array([[3, 4]]))

    created = workflow.create_representations(
        plan.missing, representation_strategy=Strategy(), combo_cfg=SimpleNamespace(),
        representation_cache_params={}, feature_level="patch",
    )
    loaded = workflow.load_representations(plan.existing)
    assembled = workflow.assemble_representations(
        plan, loaded=loaded, created=created, aggregation_level="patient", exclusion_level="patient",
        build_exclusion_key=lambda **kwargs: kwargs["sample"].patient_id,
    )
    assert created_samples == ["b"]
    result = assembled["reference"][0]
    np.testing.assert_array_equal(result.data, [[3, 4], [1, 2]])
    assert result.feature_level == "patch"
    assert result.exclusion_key == "patient"
    assert result.additional_data["source_slide_id"].tolist() == ["b", "a"]
    assert result is not assembled["query"][0]
    assert all("dataset_name" not in rep.additional_data for rep in loaded.values())
    next_plan = workflow.plan_representations(datasets_by_use={"reference": [dataset]}, representation_id="rep")
    assert not next_plan.missing
    np.testing.assert_array_equal(workflow.load_representations(next_plan.existing)[next(iter(created))].data, [[3, 4]])


def test_slide_assembly_uses_fresh_runtime_metadata_for_each_use(tmp_path):
    save_cached(tmp_path, "a", [[1, 2]])
    plan = workflow.plan_representations(
        datasets_by_use={
            "reference": [Dataset(tmp_path, [sample(tmp_path, "ref", ["a"], "p1")], "ref")],
            "query": [Dataset(tmp_path, [sample(tmp_path, "query", ["a"], "p2")], "query")],
        }, representation_id="rep",
    )
    loaded = workflow.load_representations(plan.existing)
    result = workflow.assemble_representations(
        plan, loaded=loaded, created={}, aggregation_level="slide", exclusion_level="patient",
        build_exclusion_key=lambda **kwargs: kwargs["sample"].patient_id,
    )
    assert result["reference"][0].sample_id == "ref"
    assert result["query"][0].sample_id == "query"
    assert result["reference"][0].metadata.patient_id == "p1"
    assert result["query"][0].metadata.patient_id == "p2"
    assert next(iter(loaded.values())).metadata.patient_id is None


@pytest.mark.parametrize("datasets,match", [({"train": []}, "Unsupported"), ({"query": [object()]}, "requires SlideRetrievalBagDataset")])
def test_plan_rejects_invalid_context(datasets, match):
    with pytest.raises((ValueError, TypeError), match=match):
        workflow.plan_representations(datasets_by_use=datasets, representation_id="rep")


@pytest.mark.parametrize("members,paths", [([], []), (["a"], [])])
def test_plan_rejects_invalid_member_lists(tmp_path, members, paths):
    group = sample(tmp_path, "group", members)
    group.artifact_paths = paths
    with pytest.raises(ValueError, match="Invalid physical-slide members"):
        workflow.plan_representations(datasets_by_use={"query": [Dataset(tmp_path, [group])]}, representation_id="rep")


def test_empty_workflow_has_no_side_effects():
    plan = workflow.plan_representations(datasets_by_use={"query": []}, representation_id="rep")
    assert workflow.load_representations({}) == {}
    assert workflow.create_representations(
        {}, representation_strategy=None, combo_cfg=None, representation_cache_params={}, feature_level="patch"
    ) == {}
    assert workflow.assemble_representations(
        plan, loaded={}, created={}, aggregation_level="slide", exclusion_level="none", build_exclusion_key=lambda **_: None
    ) == {"query": []}


def test_load_fails_if_planned_cache_entry_disappears(tmp_path):
    save_cached(tmp_path, "a", [[1, 2]])
    plan = workflow.plan_representations(datasets_by_use={"query": [Dataset(tmp_path, [sample(tmp_path, "a", ["a"])])]}, representation_id="rep")
    key = next(iter(plan.existing))
    with FileHandleH5(key[0], mode="a") as artifact:
        del artifact.h5["bags"]
    with pytest.raises(RuntimeError, match="disappeared"):
        workflow.load_representations(plan.existing)


def test_create_wraps_strategy_failure_with_slide_identity(tmp_path):
    plan = workflow.plan_representations(datasets_by_use={"query": [Dataset(tmp_path, [sample(tmp_path, "a", ["a"])])]}, representation_id="rep")
    with pytest.raises(RuntimeError, match="creation failed for sample 'a'"):
        workflow.create_representations(
            plan.missing, representation_strategy=SimpleNamespace(load_sample=lambda **_: (_ for _ in ()).throw(ValueError("failed"))),
            combo_cfg=None, representation_cache_params={}, feature_level="patch",
        )
    assert not next(iter(plan.missing))[0].exists()


@pytest.mark.parametrize("overlap", [False, True])
def test_assembly_rejects_incomplete_or_overlapping_results(tmp_path, overlap):
    plan = workflow.plan_representations(datasets_by_use={"query": [Dataset(tmp_path, [sample(tmp_path, "a", ["a"])])]}, representation_id="rep")
    results = {key: RetrievalRepresentation(sample_id="a", data=[[1, 2]]) for key in plan.missing} if overlap else {}
    with pytest.raises(ValueError):
        workflow.assemble_representations(
            plan, loaded=results, created=results, aggregation_level="slide", exclusion_level="none", build_exclusion_key=lambda **_: None
        )


def test_plan_carries_physical_slide_annotation_snapshot(tmp_path):
    import pandas as pd

    group = sample(tmp_path, "patient", ["S1", "S2"])
    group.metadata = {"dataset": "dataset"}
    group.annotations_df = pd.DataFrame([
        {"dataset": "dataset", "slide": "S1", "fallback_mpp": 0.25},
        {"dataset": "dataset", "slide": "S2", "fallback_mpp": 0.5},
    ])
    plan = workflow.plan_representations(
        datasets_by_use={"reference": [Dataset(tmp_path, [group])]},
        representation_id="rep",
    )
    from pathforge.slide_retrieval.annotations import resolve_sample_annotations

    for request in plan.missing.values():
        rows = resolve_sample_annotations(sample=request.sample, config={})
        assert rows[request.sample.sample_id]["fallback_mpp"] == (
            0.25 if request.sample.sample_id == "S1" else 0.5
        )
