"""Smoke test for the end-to-end MIL pipeline using mock features."""

from __future__ import annotations

import pandas as pd
import torch

from pathbench.config.config import Config
from pathbench.core.models.mil_base import MILModelBase
from pathbench.policy.benchmarking import BenchmarkingPolicy
from pathbench.policy.optimization import OptimizationPolicy
from pathbench.utils.registries import MODELS


class SmokeGaussianMIL(MILModelBase):
    """Lightweight MIL model for smoke-testing the pipeline."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)

    @property
    def bag_size(self) -> int | None:
        return None

    def forward_bag(self, bag: torch.Tensor, mask: torch.Tensor | None = None, **_: object) -> torch.Tensor:
        if mask is not None:
            masked = bag * mask.unsqueeze(-1)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
            pooled = masked.sum(dim=1) / denom
        else:
            pooled = bag.mean(dim=1)
        return self.fc(pooled)


def _register_smoke_model(name: str) -> None:
    if MODELS.is_available(name):
        return
    MODELS.register(name)(SmokeGaussianMIL)


def test_mock_pipeline_smoke(tmp_path, monkeypatch):
    """
    Run benchmarking and optimization on a tiny synthetic dataset.

    Expected bag shape: (num_tiles=10, feature_dim=6).
    """
    _register_smoke_model("SmokeGaussianMIL")

    features_dir = tmp_path / "features"
    features_dir.mkdir()

    slide_ids = [f"slide_{i}" for i in range(4)]
    for idx, slide_id in enumerate(slide_ids):
        bag = torch.randn(10, 6) + (0.25 if idx % 2 else -0.25)
        torch.save(bag, features_dir / f"{slide_id}.pt")

    annotations = pd.DataFrame(
        {
            "slide_id": slide_ids,
            "dataset": ["train", "train", "val", "val"],
            "label": [0, 1, 0, 1],
        }
    )
    ann_path = tmp_path / "annotations.csv"
    annotations.to_csv(ann_path, index=False)

    config = Config.from_dict(
        {
            "experiment": {
                "project_name": "smoke",
                "annotation_file": str(ann_path),
                "task": "classification",
                "mode": "benchmark",
                "project_root": str(tmp_path),
                "trainer_backend": "simple",
            },
            "mil": {"epochs": 1, "batch_size": 2, "k": 2},
            "bag_dataset": {
                "id_column": "slide_id",
                "label_column": "label",
                "dataset_column": "dataset",
                "label_dtype": "int",
            },
            "evaluation": {"metrics": ["accuracy"]},
            "optimization": {
                "study_name": "smoke_study",
                "trials": 1,
                "objective_metric": "val_loss",
                "objective_mode": "min",
            },
            "datasets": [
                {
                    "name": "train",
                    "slide_path": str(tmp_path),
                    "features_path": str(features_dir),
                    "used_for": "training",
                },
                {
                    "name": "val",
                    "slide_path": str(tmp_path),
                    "features_path": str(features_dir),
                    "used_for": "validation",
                },
            ],
            "search_space": {"mil": ["SmokeGaussianMIL"], "loss": ["CrossEntropyLoss"]},
        }
    )

    monkeypatch.chdir(tmp_path)
    BenchmarkingPolicy(config).execute()
    assert (tmp_path / "benchmark_results.csv").exists()

    OptimizationPolicy(config).execute()
    assert (tmp_path / "smoke_study_results.csv").exists()