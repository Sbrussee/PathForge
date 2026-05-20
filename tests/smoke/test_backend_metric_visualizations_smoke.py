"""Smoke coverage for backend-specific training metric visualizations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from pathbench.adapters.mil_lab.backend import MILLabBackendModel, register_mil_lab_backend
from pathbench.adapters.torchmil.backend import (
    TorchMILBackendModel,
    register_torchmil_backend,
)
from pathbench.core.datasets.bag_dataset import BagDataset
from pathbench.training.lightning import LightningTrainer
from pathbench.utils.registries import LOSSES

from ._smoke_dataset import PreparedBagWorkspace, attach_smoke_outputs, capture_smoke_metrics
from ._smoke_training import (
    DEFAULT_SMOKE_EPOCHS,
    SmokeTrainingResult,
    fit_smoke_model,
    make_training_config,
    training_artifact_outputs,
)


class _FakeTorchMILModel(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.head = torch.nn.Linear(input_dim, 2, bias=False)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[0, 0] = 1.0
            self.head.weight[1, 1] = 1.0

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pooled = batch["X"].float().mean(dim=1)
        return {"logits": self.head(pooled)}


class _FakeMILLabModel(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.head = torch.nn.Linear(input_dim, 2, bias=False)
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
        pooled = bag.float().mean(dim=1)
        logits = self.head(pooled)
        return {"logits": logits}, {"ignored": True}


def _make_dataset(workspace: PreparedBagWorkspace) -> BagDataset:
    return BagDataset(
        "backend_visualization_smoke",
        str(workspace.feature_dir),
        str(workspace.metadata_csv),
        "binary_label",
    )


@pytest.mark.smoke
def test_torchmil_backend_classification_visualizations_smoke(
    extracted_bag_workspace: PreparedBagWorkspace,
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
        lambda spec, config_kwargs: _FakeTorchMILModel(config_kwargs["in_shape"][0]),
    )

    dataset = _make_dataset(extracted_bag_workspace)
    cfg = make_training_config(
        tmp_path / "torchmil_backend",
        task="classification",
        epochs=DEFAULT_SMOKE_EPOCHS,
        lr=1e-3,
        dropout=0.0,
        batch_size=1,
    )
    cfg.mil.backend = "torchmil"
    cfg.mil.torchmil_model = "ABMIL"
    cfg.mil.torchmil_model_kwargs = {"in_shape": (extracted_bag_workspace.input_dim,), "out_shape": 2}
    cfg.mil.best_epoch_based_on = "balanced_accuracy"
    cfg.mil.use_torchmil_collate = False

    model = TorchMILBackendModel(
        torchmil_model="ABMIL",
        task="classification",
        torchmil_model_kwargs={"in_shape": (extracted_bag_workspace.input_dim,), "out_shape": 2},
    )
    trainer = LightningTrainer(cfg)
    loss_fn = LOSSES.get("CrossEntropyLoss")()

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="hf_torchmil_backend_visualizations",
        metadata={"num_bags": len(dataset)},
    ) as metadata:
        best_model_path, _ = trainer.fit(model, dataset, dataset, loss_fn)
        artifacts_dir = Path(cfg.experiment.project_root) / "training_artifacts"
        attach_smoke_outputs(
            metadata,
            step_name="hf_torchmil_backend_visualizations",
            intermediate={"bag_metadata_csv": extracted_bag_workspace.metadata_csv},
            final={
                "best_model_path": Path(best_model_path),
                **training_artifact_outputs(
                    SmokeTrainingResult(
                        best_model_path=str(best_model_path),
                        best_score=0.0,
                        output_dim=2,
                        task_name="classification",
                        artifacts_dir=str(artifacts_dir),
                    )
                ),
            },
        )

    assert Path(best_model_path).exists()
    assert (artifacts_dir / "val_confusion_matrix.png").exists()
    assert (artifacts_dir / "val_roc_auc_curve.png").exists()
    assert (artifacts_dir / "val_pr_auc_curve.png").exists()
    assert (artifacts_dir / "val_calibration_curve.png").exists()


@pytest.mark.smoke
def test_multiclass_classification_heatmap_smoke(
    extracted_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
) -> None:
    """Verify confusion-matrix heatmap and curves are generated for 3+ classes."""
    metadata_df = pd.read_csv(extracted_bag_workspace.metadata_csv)
    num_classes = int(metadata_df["multiclass_label"].nunique())
    if num_classes < 3:
        pytest.skip("Fewer than 3 multiclass labels; skipping multiclass smoke test.")

    dataset = BagDataset(
        "multiclass_heatmap_smoke",
        str(extracted_bag_workspace.feature_dir),
        str(extracted_bag_workspace.metadata_csv),
        "multiclass_label",
    )

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="hf_multiclass_heatmap_smoke",
        metadata={"num_classes": num_classes, "num_bags": len(dataset)},
    ) as metadata:
        _, result = fit_smoke_model(
            tmp_path / "multiclass_heatmap",
            dataset_train=dataset,
            dataset_val=dataset,
            input_dim=extracted_bag_workspace.input_dim,
            output_dim=num_classes,
            task="classification",
            loss_name="CrossEntropyLoss",
            epochs=DEFAULT_SMOKE_EPOCHS,
            lr=1e-3,
            dropout=0.0,
        )
        artifacts_dir = Path(result.artifacts_dir)
        attach_smoke_outputs(
            metadata,
            step_name="hf_multiclass_heatmap_smoke",
            final={
                **training_artifact_outputs(result),
            },
        )

    assert (artifacts_dir / "val_confusion_matrix.png").exists(), (
        "Confusion matrix heatmap not generated for multiclass task."
    )
    assert (artifacts_dir / "val_roc_auc_curve.png").exists()
    assert (artifacts_dir / "val_pr_auc_curve.png").exists()
    assert (artifacts_dir / "val_calibration_curve.png").exists()


@pytest.mark.smoke
def test_mil_lab_backend_classification_visualizations_smoke(
    extracted_bag_workspace: PreparedBagWorkspace,
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
        lambda spec, config_kwargs, from_pretrained=False: _FakeMILLabModel(
            int(config_kwargs["input_dim"])
        ),
    )

    dataset = _make_dataset(extracted_bag_workspace)
    cfg = make_training_config(
        tmp_path / "mil_lab_backend",
        task="classification",
        epochs=DEFAULT_SMOKE_EPOCHS,
        lr=1e-3,
        dropout=0.0,
        batch_size=1,
    )
    cfg.mil.backend = "mil-lab"
    cfg.mil.mil_lab_model = "abmil"
    cfg.mil.mil_lab_model_kwargs = {"input_dim": extracted_bag_workspace.input_dim, "output_dim": 2}
    cfg.mil.best_epoch_based_on = "balanced_accuracy"

    model = MILLabBackendModel(
        mil_lab_model="abmil",
        task="classification",
        mil_lab_model_kwargs={"input_dim": extracted_bag_workspace.input_dim, "output_dim": 2},
    )
    trainer = LightningTrainer(cfg)
    loss_fn = LOSSES.get("CrossEntropyLoss")()

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="hf_mil_lab_backend_visualizations",
        metadata={"num_bags": len(dataset)},
    ) as metadata:
        best_model_path, _ = trainer.fit(model, dataset, dataset, loss_fn)
        artifacts_dir = Path(cfg.experiment.project_root) / "training_artifacts"
        attach_smoke_outputs(
            metadata,
            step_name="hf_mil_lab_backend_visualizations",
            intermediate={"bag_metadata_csv": extracted_bag_workspace.metadata_csv},
            final={
                "best_model_path": Path(best_model_path),
                **training_artifact_outputs(
                    SmokeTrainingResult(
                        best_model_path=str(best_model_path),
                        best_score=0.0,
                        output_dim=2,
                        task_name="classification",
                        artifacts_dir=str(artifacts_dir),
                    )
                ),
            },
        )

    assert Path(best_model_path).exists()
    assert (artifacts_dir / "val_confusion_matrix.png").exists()
    assert (artifacts_dir / "val_roc_auc_curve.png").exists()
    assert (artifacts_dir / "val_pr_auc_curve.png").exists()
    assert (artifacts_dir / "val_calibration_curve.png").exists()
