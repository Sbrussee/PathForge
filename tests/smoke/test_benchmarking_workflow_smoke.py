from __future__ import annotations

from types import SimpleNamespace

import pytest

from pathbench.core.experiments.combinations import ComboConfig
from pathbench.policy.benchmarking import BenchmarkingPolicy


class _FakeTask:
    @classmethod
    def get_grid_keys(cls) -> list[str]:
        return ["feature_extraction", "tile_px", "tile_mpp", "mil"]

    def execute(self, combo_cfg: ComboConfig, datasets_by_use: dict[str, list[object]]) -> dict[str, object]:
        return {"combo_cfg": combo_cfg, "datasets_by_use": datasets_by_use}


@pytest.mark.smoke
def test_smoke_benchmarking_policy_executes_single_combo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_task = _FakeTask()
    policy = BenchmarkingPolicy.__new__(BenchmarkingPolicy)
    policy.task_name = "slide_retrieval"
    policy.task = fake_task
    policy.experiment = SimpleNamespace(load_annotations=lambda: SimpleNamespace(empty=False))
    policy.cfg = SimpleNamespace()
    policy.ensure_bag_features_exist = lambda **kwargs: None  # type: ignore[method-assign]
    policy.build_bag_datasets_for_combo = lambda **kwargs: [SimpleNamespace(name="bag")]  # type: ignore[method-assign]
    policy.group_bag_datasets_by_use = lambda datasets: {"reference": datasets}  # type: ignore[method-assign]
    policy._validate_dataset_uses = lambda datasets_by_use: None  # type: ignore[method-assign]

    combo = ComboConfig(
        feature_extraction="uni",
        tile_px=256,
        tile_mpp=0.5,
        mil="attention_mil",
    )

    out = policy.execute_combination(combo)

    assert out["status"] == "benchmark_done"
    assert out["num_runs"] == 1
    assert out["task_output"]["datasets_by_use"]["reference"][0].name == "bag"
