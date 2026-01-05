import pandas as pd
import pytest
import torch

from pathbench.config.config import Config
from pathbench.core.datasets.bag_dataset import BagDataset
from pathbench.core.models.mil_base import MILModelBase
from pathbench.core.losses.classification import CrossEntropyLoss
from pathbench.core.losses.regression import MSELoss
from pathbench.core.losses.survival_continuous import CoxPHLoss
from pathbench.core.losses.survival_discrete import DiscreteTimeNLLLoss
from pathbench.training.metrics import evaluate_predictions
from pathbench.training.simple import SimpleTrainer


class MeanPoolMIL(MILModelBase):
    """Minimal MIL model for pipeline testing."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.encoder = torch.nn.Linear(input_dim, output_dim)

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
        return self.encoder(pooled)


@pytest.mark.parametrize(
    "task,output_dim,loss_cls,metrics,label_dtype",
    [
        ("classification", 2, CrossEntropyLoss, ["accuracy"], "int"),
        ("classification", 3, CrossEntropyLoss, ["accuracy"], "int"),
        ("regression", 1, MSELoss, ["mse"], "float"),
        ("survival", 1, CoxPHLoss, ["c_index"], "float"),
        ("survival_discrete", 4, DiscreteTimeNLLLoss, ["c_index"], "float"),
    ],
)
def test_simple_trainer_fit_predict(tmp_path, task, output_dim, loss_cls, metrics, label_dtype):
    if task in {"survival", "survival_discrete"}:
        pytest.importorskip("pycox")
    feature_dim = 8
    slide_ids = [f"slide_{i}" for i in range(4)]
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    for slide_id in slide_ids:
        torch.save(torch.randn(4, feature_dim), features_dir / f"{slide_id}.pt")

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
                "project_name": "unit",
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
                "label_dtype": label_dtype,
                "time_column": "time" if task in {"survival", "survival_discrete"} else None,
                "event_column": "event" if task in {"survival", "survival_discrete"} else None,
            },
            "evaluation": {"metrics": metrics},
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
        }
    )

    train_ds = BagDataset.from_config(config.datasets[0], config)
    val_ds = BagDataset.from_config(config.datasets[1], config)

    model = MeanPoolMIL(input_dim=feature_dim, output_dim=output_dim)
    loss_fn = loss_cls()
    trainer = SimpleTrainer(config)

    result = trainer.fit(model, train_ds, val_ds, loss_fn)
    assert result.best_model_path

    preds = trainer.predict(model, val_ds)
    assert preds.shape[0] == len(val_ds)

    metrics_out = evaluate_predictions(
        preds,
        val_ds.labels,
        task,
        metrics=config.evaluation.metrics,
        average=config.evaluation.average,
        positive_label=config.evaluation.positive_label,
    )
    assert metrics_out