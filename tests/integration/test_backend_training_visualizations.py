from __future__ import annotations

from pathlib import Path

import torch

from pathbench.adapters.mil_lab.backend import MILLabBackendModel, register_mil_lab_backend
from pathbench.adapters.torchmil.backend import (
    TorchMILBackendModel,
    register_torchmil_backend,
)
from pathbench.config.config import Config
from pathbench.training.lightning import LightningTrainer
from pathbench.utils.registries import LOSSES
from tests.conftest import DUMMY_FE, DUMMY_MIL


class _ToyClassificationDataset:
    def __init__(self) -> None:
        self._bags = [
            torch.tensor([[4.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([[3.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 4.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 3.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]], dtype=torch.float32),
        ]
        self._labels = [
            torch.tensor(0, dtype=torch.long),
            torch.tensor(0, dtype=torch.long),
            torch.tensor(1, dtype=torch.long),
            torch.tensor(1, dtype=torch.long),
        ]

    def __len__(self) -> int:
        return len(self._bags)

    def __getitem__(self, index: int):
        return {"X": self._bags[index], "Y": self._labels[index]}


class _FakeTorchMILModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[0, 0] = 1.0
            self.head.weight[1, 1] = 1.0

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        bag = batch["X"].float()
        pooled = bag.mean(dim=1)
        logits = self.head(pooled)
        return {"logits": logits}


class _FakeMILLabModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[0, 0] = 1.0
            self.head.weight[1, 1] = 1.0

    def forward(
        self,
        bag: torch.Tensor,
        *,
        loss_fn=None,
        label=None,
        return_attention: bool = False,
        return_slide_feats: bool = False,
    ):
        _ = (loss_fn, label, return_attention, return_slide_feats)
        pooled = bag.mean(dim=1)
        logits = self.head(pooled)
        return {"logits": logits}, {"ignored": True}


def _make_cfg(tmp_path: Path) -> Config:
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text("dataset,slide,patient,category\n", encoding="utf-8")
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    return Config.model_validate(
        {
            "experiment": {
                "project_name": "backend_viz",
                "annotation_file": str(annotation_path),
                "project_root": str((tmp_path / "project").resolve()),
                "mode": "benchmark",
                "task": "classification",
                "num_workers": 0,
            },
            "mil": {
                "backend": "native",
                "epochs": 1,
                "batch_size": 2,
                "patience": 1,
                "best_epoch_based_on": "balanced_accuracy",
            },
            "metrics": {"classification_backend": "native"},
            "slide_processing": {"backend": "lazyslide"},
            "datasets": [
                {
                    "name": "backend_ds",
                    "slides_dir": str(slides_dir),
                    "artifacts_dir": str(tmp_path / "artifacts"),
                    "used_for": "all",
                }
            ],
            "benchmark_parameters": {
                "tile_px": [256],
                "tile_mpp": [0.5],
                "feature_extraction": [DUMMY_FE],
                "mil": [DUMMY_MIL],
                "loss": ["CrossEntropyLoss"],
            },
        }
    )


def test_torchmil_backend_trainer_writes_classification_visualizations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    register_torchmil_backend()
    monkeypatch.setattr(
        "pathbench.adapters.torchmil.backend.require_torchmil",
        lambda feature: None,
    )
    monkeypatch.setattr(
        "pathbench.adapters.torchmil.backend.build_torchmil_model",
        lambda spec, config_kwargs: _FakeTorchMILModel(),
    )

    cfg = _make_cfg(tmp_path)
    cfg.mil.backend = "torchmil"
    cfg.mil.torchmil_model = "ABMIL"
    cfg.mil.torchmil_model_kwargs = {"in_shape": (4,), "out_shape": 2}

    model = TorchMILBackendModel(
        torchmil_model="ABMIL",
        task="classification",
        torchmil_model_kwargs={"in_shape": (4,), "out_shape": 2},
    )
    trainer = LightningTrainer(cfg)
    loss_fn = LOSSES.get("CrossEntropyLoss")()
    dataset = _ToyClassificationDataset()

    best_model_path, _ = trainer.fit(model, dataset, dataset, loss_fn)

    artifacts_dir = Path(cfg.experiment.project_root) / "training_artifacts"
    assert Path(best_model_path).exists()
    assert (artifacts_dir / "val_metrics.json").exists()
    assert (artifacts_dir / "val_confusion_matrix.png").exists()
    assert (artifacts_dir / "val_roc_auc_curve.png").exists()
    assert (artifacts_dir / "val_pr_auc_curve.png").exists()
    assert (artifacts_dir / "val_calibration_curve.png").exists()


def test_mil_lab_backend_trainer_writes_classification_visualizations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    register_mil_lab_backend()
    monkeypatch.setattr(
        "pathbench.adapters.mil_lab.backend.require_mil_lab",
        lambda feature: None,
    )
    monkeypatch.setattr(
        "pathbench.adapters.mil_lab.backend.build_mil_lab_model",
        lambda spec, config_kwargs, from_pretrained=False: _FakeMILLabModel(),
    )

    cfg = _make_cfg(tmp_path)
    cfg.mil.backend = "mil-lab"
    cfg.mil.mil_lab_model = "abmil"
    cfg.mil.mil_lab_model_kwargs = {"input_dim": 4, "output_dim": 2}

    model = MILLabBackendModel(
        mil_lab_model="abmil",
        task="classification",
        mil_lab_model_kwargs={"input_dim": 4, "output_dim": 2},
    )
    trainer = LightningTrainer(cfg)
    loss_fn = LOSSES.get("CrossEntropyLoss")()
    dataset = _ToyClassificationDataset()

    best_model_path, _ = trainer.fit(model, dataset, dataset, loss_fn)

    artifacts_dir = Path(cfg.experiment.project_root) / "training_artifacts"
    assert Path(best_model_path).exists()
    assert (artifacts_dir / "val_metrics.json").exists()
    assert (artifacts_dir / "val_confusion_matrix.png").exists()
    assert (artifacts_dir / "val_roc_auc_curve.png").exists()
    assert (artifacts_dir / "val_pr_auc_curve.png").exists()
    assert (artifacts_dir / "val_calibration_curve.png").exists()
