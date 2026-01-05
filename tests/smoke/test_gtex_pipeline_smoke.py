import pandas as pd
import pytest
import torch

from pathbench.config.config import Config
from pathbench.core.losses import (
    classification,  # noqa: F401
    regression,  # noqa: F401
    survival_continuous,  # noqa: F401
    survival_discrete,  # noqa: F401
)
from pathbench.core.models.mil_base import MILModelBase
from pathbench.policy.benchmarking import BenchmarkingPolicy
from pathbench.policy.optimization import OptimizationPolicy
from pathbench.utils.registries import MODELS
from pathbench.utils.test_samples import download_gtex_slides


"""

This smoke tests should run an end-to-end GTEx pipeline for all supported tasks with minimal settings.

"""

class SmokeMIL(MILModelBase):
    """Simple MIL model for smoke tests."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

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


def _register_smoke_model(name: str) -> None:
    if MODELS.is_available(name):
        return

    @MODELS.register(name)
    class _RegisteredSmokeMIL(SmokeMIL):
        pass


@pytest.fixture(scope="session")
def gtex_slides() -> list[str]:
    pytest.importorskip("huggingface_hub")
    try:
        gtex = download_gtex_slides()
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"Failed to download GTEx sample data: {exc}")

    slide_col = next(
        (col for col in ("slide_id", "slide", "slide_name") if col in gtex.columns),
        gtex.columns[0],
    )
    return gtex[slide_col].astype(str).head(6).tolist()


@pytest.fixture(scope="session")
def gtex_features(tmp_path_factory, gtex_slides: list[str]) -> str:
    """
    Simulate feature extraction by materializing random bags for GTEx slides.
    """
    features_dir = tmp_path_factory.mktemp("gtex_features")
    for slide_id in gtex_slides:
        torch.save(torch.randn(4, 8), features_dir / f"{slide_id}.pt")
    return str(features_dir)


def _build_annotations(slides: list[str], labels: list, extra: dict[str, list]) -> pd.DataFrame:
    data = {
        "slide_id": slides,
        "dataset": ["train"] * (len(slides) // 2) + ["val"] * (len(slides) - len(slides) // 2),
        "label": labels,
    }
    data.update(extra)
    return pd.DataFrame(data)


def _build_config(
    tmp_path: str,
    annotations_path: str,
    features_dir: str,
    model_name: str,
    task: str,
    output_dim: int,
    metrics: list[str],
    loss_name: str,
) -> Config:
    return Config.from_dict(
        {
            "experiment": {
                "project_name": "gtex",
                "annotation_file": annotations_path,
                "task": task,
                "mode": "benchmark",
                "project_root": tmp_path,
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
                "study_name": f"gtex_study_{task}",
                "trials": 1,
                "objective_metric": "val_loss",
                "objective_mode": "min",
            },
            "datasets": [
                {
                    "name": "train",
                    "slide_path": tmp_path,
                    "features_path": features_dir,
                    "used_for": "training",
                },
                {
                    "name": "val",
                    "slide_path": tmp_path,
                    "features_path": features_dir,
                    "used_for": "validation",
                },
            ],
            "search_space": {"mil": [model_name], "loss": [loss_name]},
        }
    )


def test_gtex_pipeline_smoke(tmp_path, monkeypatch, gtex_slides, gtex_features):
    _register_smoke_model("SmokeMIL")

    tasks = [
        ("classification", 2, [0, 1, 0, 1, 0, 1], {}, ["accuracy"], "CrossEntropyLoss"),
        ("classification", 3, [0, 1, 2, 1, 0, 2], {}, ["accuracy"], "CrossEntropyLoss"),
        ("regression", 1, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], {}, ["mse"], "MSELoss"),
        ("survival", 1, [0] * 6, {"time": [1, 2, 3, 4, 5, 6], "event": [1, 0, 1, 1, 0, 1]}, ["c_index"], "CoxPHLoss"),
        (
            "survival_discrete",
            4,
            [0] * 6,
            {"time": [0, 1, 2, 3, 1, 2], "event": [1, 0, 1, 1, 0, 1]},
            ["c_index"],
            "DiscreteTimeNLLLoss",
        ),
    ]

    for task, output_dim, labels, extra, metrics, loss_name in tasks:
        if task in {"survival", "survival_discrete"}:
            pytest.importorskip("pycox")

        annotations = _build_annotations(gtex_slides, labels, extra)
        ann_path = tmp_path / f"annotations_{task}.csv"
        annotations.to_csv(ann_path, index=False)

        config = _build_config(
            tmp_path=str(tmp_path),
            annotations_path=str(ann_path),
            features_dir=gtex_features,
            model_name="SmokeMIL",
            task=task,
            output_dim=output_dim,
            metrics=metrics,
            loss_name=loss_name,
        )

        monkeypatch.chdir(tmp_path)
        BenchmarkingPolicy(config).execute()
        assert (tmp_path / "benchmark_results.csv").exists()

        OptimizationPolicy(config).execute()
        assert (tmp_path / f"gtex_study_{task}_results.csv").exists()