"""Unit tests for MIL training on synthetic Gaussian-distributed features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from pathbench.config.config import Config
from pathbench.core.datasets.bag_dataset import BagDataset
from pathbench.core.losses.classification import CrossEntropyLoss
from pathbench.core.models.mil_base import MILModelBase
from pathbench.training.simple import SimpleTrainer
from pathbench.utils.registries import MODELS


class MeanPoolGaussianMIL(MILModelBase):
    """Minimal MIL model that mean-pools instances before classification."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.encoder = torch.nn.Linear(input_dim, output_dim)

    @property
    def bag_size(self) -> int | None:
        return None

    def forward_bag(
        self,
        bag: torch.Tensor,
        mask: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        if mask is not None:
            masked = bag * mask.unsqueeze(-1)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
            pooled = masked.sum(dim=1) / denom
        else:
            pooled = bag.mean(dim=1)
        return self.encoder(pooled)


def _write_gaussian_bag(
    features_dir: Path,
    slide_id: str,
    num_tiles: int,
    feature_dim: int,
    mean_shift: float,
    seed: int,
) -> torch.Tensor:
    """
    Persist a Gaussian bag with shape (num_tiles, feature_dim).

    The mean_shift parameter creates overlapping but separable distributions.
    """
    generator = torch.Generator().manual_seed(seed)
    bag = torch.randn(num_tiles, feature_dim, generator=generator) + mean_shift
    torch.save(bag, features_dir / f"{slide_id}.pt")
    return bag


def _build_config(tmp_path: Path, ann_path: Path, features_dir: Path) -> Config:
    return Config.from_dict(
        {
            "experiment": {
                "project_name": "gaussian",
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
            "search_space": {"mil": ["MeanPoolGaussianMIL"], "loss": ["CrossEntropyLoss"]},
        }
    )


@pytest.mark.parametrize("feature_dim", [8, 16])
def test_gaussian_mil_training_pipeline(tmp_path, feature_dim):
    """
    Train a simple MIL model on two overlapping Gaussian distributions.

    Feature bags are stored with shapes (num_tiles, feature_dim),
    where num_tiles is sampled between 100 and 200 per slide.
    """
    torch.manual_seed(0)
    features_dir = tmp_path / "features"
    features_dir.mkdir()

    slide_specs = [
        ("slide_0", 0, -0.25),
        ("slide_1", 0, -0.25),
        ("slide_2", 1, 0.25),
        ("slide_3", 1, 0.25),
    ]

    num_tiles_by_slide: dict[str, int] = {}
    for idx, (slide_id, _label, mean_shift) in enumerate(slide_specs):
        num_tiles = int(torch.randint(100, 201, (1,)).item())
        num_tiles_by_slide[slide_id] = num_tiles
        _write_gaussian_bag(
            features_dir=features_dir,
            slide_id=slide_id,
            num_tiles=num_tiles,
            feature_dim=feature_dim,
            mean_shift=mean_shift,
            seed=idx,
        )

    annotations = pd.DataFrame(
        {
            "slide_id": [spec[0] for spec in slide_specs],
            "dataset": ["train", "train", "val", "val"],
            "label": [spec[1] for spec in slide_specs],
        }
    )
    ann_path = tmp_path / "annotations.csv"
    annotations.to_csv(ann_path, index=False)

    if not MODELS.is_available("MeanPoolGaussianMIL"):
        MODELS.register("MeanPoolGaussianMIL")(MeanPoolGaussianMIL)

    config = _build_config(tmp_path, ann_path, features_dir)
    train_ds = BagDataset.from_config(config.datasets[0], config)
    val_ds = BagDataset.from_config(config.datasets[1], config)
    assert all(100 <= count <= 200 for count in num_tiles_by_slide.values())

    model = MeanPoolGaussianMIL(input_dim=feature_dim, output_dim=2)
    trainer = SimpleTrainer(config)
    loss_fn = CrossEntropyLoss()

    result = trainer.fit(model, train_ds, val_ds, loss_fn)
    assert result.best_model_path

    preds = trainer.predict(model, val_ds)
    assert preds.shape == (len(val_ds), 2)