import pandas as pd
import pytest
import torch

from pathbench.config.config import Config
from pathbench.core.models.mil_base import MILModelBase
from pathbench.core.losses import classification, regression, survival_continuous, survival_discrete  # noqa: F401
from pathbench.policy.benchmarking import BenchmarkingPolicy
from pathbench.policy.optimization import OptimizationPolicy
from pathbench.utils.registries import MODELS


class TinyMIL(MILModelBase):
    """Minimal MIL model used for policy tests."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
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
        return self.fc(self.dropout(pooled))


def _register_model(name: str) -> None:
    if MODELS.is_available(name):
        return

    @MODELS.register(name)
    class _RegisteredTinyMIL(TinyMIL):
        pass


@pytest.mark.parametrize(
    "task,output_dim,loss_name,metrics",
    [
        ("classification", 2, "CrossEntropyLoss", ["accuracy"]),
        ("classification", 3, "CrossEntropyLoss", ["accuracy"]),
        ("regression", 1, "MSELoss", ["mse"]),
        ("survival", 1, "CoxPHLoss", ["c_index"]),
        ("survival_discrete", 4, "DiscreteTimeNLLLoss", ["c_index"]),
    ],
)
def test_benchmark_and_optimization_policies(tmp_path, monkeypatch, task, output_dim, loss_name, metrics):
    model_name = f"TinyMIL_{task}"
    _register_model(model_name)

    slide_ids = [f"slide_{i}" for i in range(4)]
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    for slide_id in slide_ids:
        torch.save(torch.randn(4, 8), features_dir / f"{slide_id}.pt")

    labels = [0, 1, 0, 1]
    if output_dim > 2:
        labels = [0, 1, 2, 1]

    data = {
        "slide_id": slide_ids,
        "dataset": ["train", "train", "val", "val"],
        "label": labels,
    }
    if task in {"survival", "survival_discrete"}:
        data["time"] = [1, 2, 3, 2]
        data["event"] = [1, 0, 1, 1]

    annotations = pd.DataFrame(data)
    ann_path = tmp_path / "annotations.csv"
    annotations.to_csv(ann_path, index=False)

    config = Config.from_dict(
        {
            "experiment": {
                "project_name": "policy",
                "annotation_file": str(ann_path),
                "task": task,
                "mode": "benchmark",
                "project_root": str(tmp_path),
                "trainer_backend": "simple",
            },
            "mil": {"epochs": 1, "batch_size": 2, "k": output_dim},
            "bag_dataset": {
                "id_column": "slide_id",
                "label_column": "label",
                "dataset_column": "dataset",
                "time_column": "time" if task in {"survival", "survival_discrete"} else None,
                "event_column": "event" if task in {"survival", "survival_discrete"} else None,
            },
            "evaluation": {"metrics": metrics},
            "optimization": {
                "study_name": f"study_{task}",
                "trials": 1,
                "objective_metric": "val_loss",
                "objective_mode": "min",
                "model_candidates": [model_name],
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
            "benchmark_parameters": {"mil": [model_name], "loss": [loss_name]},
        }
    )

    monkeypatch.chdir(tmp_path)
    BenchmarkingPolicy(config).execute()
    assert (tmp_path / "benchmark_results.csv").exists()

    OptimizationPolicy(config).execute()
    assert (tmp_path / f"{config.optimization.study_name}_results.csv").exists()