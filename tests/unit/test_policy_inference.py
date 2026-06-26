from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import pathbench.policy.inference as inference_mod
from pathbench.core.experiments.combinations import ComboConfig
from pathbench.policy.inference import InferencePolicy


class _FakeTask:
    inference_dataset_uses = frozenset({"reference", "query_reference"})
    inference_input_use = "query"

    def __init__(self) -> None:
        self.calls: list[tuple[ComboConfig, dict[str, list[object]], Path]] = []

    @classmethod
    def get_inference_grid_keys(cls) -> list[str]:
        return ["feature_extraction", "tile_px", "tile_mpp", "retrieval_representation", "search_strategy"]

    @classmethod
    def get_inference_dataset_uses(cls) -> frozenset[str]:
        return cls.inference_dataset_uses

    @classmethod
    def get_inference_input_use(cls) -> str:
        return cls.inference_input_use

    def inference(
        self,
        combo_cfg: ComboConfig,
        datasets_by_use: dict[str, list[object]],
        inference_run_root: Path,
    ) -> dict[str, object]:
        self.calls.append((combo_cfg, datasets_by_use, inference_run_root))
        return {"output_dir": str(inference_run_root / "task_output")}


class _FakeFeaturePolicy:
    def __init__(self, experiment: object) -> None:
        self.experiment = experiment
        self.calls: list[tuple[object, ComboConfig]] = []

    def execute_dataset(self, dataset: object, combo_cfg: ComboConfig) -> None:
        self.calls.append((dataset, combo_cfg))


def _dataset(name: str, used_for: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        used_for=used_for,
        slides_dir=f"/slides/{name}",
        artifacts_dir=f"/artifacts/{name}",
        model_copy=lambda update: _dataset(name, update["used_for"]),
    )


def _make_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[InferencePolicy, _FakeTask]:
    fake_task = _FakeTask()
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(
            task="slide_retrieval",
            mode="inference",
            aggregation_level="slide",
        ),
        datasets=[
            _dataset("ref_ds", "reference"),
            _dataset("shared_ds", "query_reference"),
            _dataset("ignored_ds", "ignore"),
        ],
        save_yaml=lambda path: Path(path).write_text("snapshot: true\n", encoding="utf-8"),
        benchmark_parameters=SimpleNamespace(),
    )
    annotations_df = pd.DataFrame(
        {
            "dataset": ["ref_ds", "shared_ds", "ignored_ds"],
            "slide": ["R1", "S1", "I1"],
            "patient": ["PR", "PS", "PI"],
            "case": ["CR", "CS", "CI"],
            "category": ["a", "b", "c"],
        }
    )
    experiment = SimpleNamespace(
        cfg=cfg,
        project_root=str(tmp_path / "project"),
        load_annotations=lambda: annotations_df,
    )

    monkeypatch.setattr(inference_mod, "import_task_modules", lambda: None)
    monkeypatch.setattr(inference_mod, "build_task", lambda task_name, experiment: fake_task)
    monkeypatch.setattr(inference_mod, "FeatureExtractionPolicy", _FakeFeaturePolicy)
    monkeypatch.setattr(
        inference_mod,
        "build_combinations",
        lambda cfg, keys: [
            ComboConfig(
                feature_extraction="uni",
                tile_px=256,
                tile_mpp=0.5,
                retrieval_representation="yottixel-features",
                search_strategy="yottixel",
            )
        ],
    )
    monkeypatch.setattr(
        inference_mod,
        "find_slides_with_missing_features",
        lambda **kwargs: [],
    )

    def fake_build_bag_dataset(
        *,
        ds_cfg: object,
        annotations_df: pd.DataFrame,
        combo_cfg: ComboConfig,
        aggregation_level: str,
        task: str,
        target_column: str | None = None,
        slide_ids: list[str] | None = None,
    ) -> object:
        _ = combo_cfg, aggregation_level, task, target_column, slide_ids
        return SimpleNamespace(
            name=f"{ds_cfg.name}_{ds_cfg.used_for}",
            ds_cfg=ds_cfg,
            slides=annotations_df["slide"].astype(str).tolist(),
        )

    monkeypatch.setattr(inference_mod, "build_bag_dataset", fake_build_bag_dataset)
    return InferencePolicy(experiment), fake_task


def test_inference_policy_uses_reference_roles_and_csv_selected_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("dataset,slide\nignored_ds,I1\n", encoding="utf-8")
    policy, fake_task = _make_policy(tmp_path, monkeypatch)

    output = policy.execute(input_csv=input_csv)

    assert output["status"] == "inference_done"
    assert len(fake_task.calls) == 1

    _, datasets_by_use, inference_run_root = fake_task.calls[0]
    assert set(datasets_by_use) == {"reference", "query_reference", "query"}
    assert datasets_by_use["reference"][0].slides == ["R1"]
    assert datasets_by_use["query_reference"][0].slides == ["S1"]
    assert datasets_by_use["query"][0].slides == ["I1"]
    assert datasets_by_use["query"][0].ds_cfg.used_for == "query"
    assert (inference_run_root / "inference_input.csv").is_file()
    assert (inference_run_root / "config_snapshot.yaml").is_file()
    assert (inference_run_root / "manifest.json").is_file()


def test_inference_policy_rejects_unknown_input_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("dataset,slide\nmissing_ds,S1\n", encoding="utf-8")
    policy, _ = _make_policy(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="unknown dataset"):
        policy.execute(input_csv=input_csv)
