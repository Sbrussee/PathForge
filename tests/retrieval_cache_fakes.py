"""In-memory cache fixtures for tests of search routing and output writing.

Real planning, creation, and HDF5 reuse are covered in test_representation_workflow.
"""

from pathlib import Path

import pathforge.core.tasks.slide_retrieval as task_module
from pathforge.core.datasets.bag_dataset import BagSample
from pathforge.slide_retrieval.representation_workflow import (
    RepresentationPlan,
    RetrievalGroup,
    SlideRepresentationRequest,
)


def stub_representation_cache(monkeypatch, collector):
    cached = {}

    def plan(*, datasets_by_use, representation_id):
        existing, missing, groups_by_use = {}, {}, {}
        cached.clear()
        for use, datasets in datasets_by_use.items():
            if use not in {"reference", "query", "query_reference"}:
                raise ValueError(f"Unsupported retrieval dataset use {use!r}.")
            groups = groups_by_use.setdefault(use, [])
            for dataset in datasets:
                reps, absent = collector(
                    None, bag_dataset=dataset, representation_id=representation_id,
                    aggregation_level=dataset.aggregation_level, exclusion_level="none",
                )
                reps_by_id = {rep.sample_id: rep for rep in reps}
                if hasattr(dataset, "get_sample"):
                    samples = [dataset.get_sample(i) for i in range(dataset.num_bags)]
                else:
                    samples = [BagSample(sample_id=dataset.sample_id, slide_ids=[dataset.sample_id], artifact_paths=[Path(f"{dataset.sample_id}.h5")], category=None)]
                for sample in samples:
                    rep = reps_by_id.get(sample.sample_id)
                    if rep is None and absent is None:
                        continue
                    keys = []
                    for slide_id, source in zip(sample.slide_ids, sample.artifact_paths):
                        key = (Path(getattr(dataset, "artifacts_dir", dataset.name)) / f"{slide_id}.h5", dataset.tiling_id, representation_id)
                        keys.append(key)
                        request = SlideRepresentationRequest(key, dataset, BagSample(
                            sample_id=str(slide_id), slide_ids=[str(slide_id)], artifact_paths=[source],
                            patient_id=sample.patient_id, case_id=sample.case_id, category=sample.category,
                        ))
                        if rep is not None:
                            if rep.feature_level is None:
                                rep.feature_level = dataset.get_feature_level()
                            existing[key] = request
                            cached[key] = rep
                        else:
                            missing[key] = request
                    groups.append(RetrievalGroup(dataset.name, sample, tuple(keys)))
        return RepresentationPlan(existing, missing, groups_by_use)

    monkeypatch.setattr(task_module, "plan_representations", plan)
    monkeypatch.setattr(task_module, "load_representations", lambda requests: {key: cached[key] for key in requests})


def stub_representation_creation(monkeypatch, creator):
    def create(requests, *, representation_strategy, combo_cfg, representation_cache_params, feature_level):
        results = {}
        for key, request in requests.items():
            try:
                result = creator(
                    None, dataset=request.dataset, sample=request.sample,
                    representation_strategy=representation_strategy, representation_id=key[2],
                    combo_cfg=combo_cfg, representation_cache_params=representation_cache_params,
                    feature_level=feature_level,
                )
            except Exception as exc:
                raise RuntimeError(f"Slide retrieval representation creation failed for sample {request.sample.sample_id!r}: {exc}") from exc
            result.feature_level = feature_level
            results[key] = result
        return results

    monkeypatch.setattr(task_module, "create_representations", create)
