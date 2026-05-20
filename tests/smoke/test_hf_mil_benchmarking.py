"""Smoke benchmarking workflows backed by reusable extracted bags."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ._smoke_dataset import (
    ExtractedWsiWorkspace,
    PreparedBagWorkspace,
    attach_smoke_outputs,
    capture_smoke_metrics,
)
from ._smoke_training import DEFAULT_SMOKE_EPOCHS, SmokeTrainingResult, fit_smoke_model
from ._smoke_training import make_training_config, training_artifact_outputs


def _build_bag_dataset(workspace: PreparedBagWorkspace, *, target_column: str):
    """Construct a production ``BagDataset`` for one target column."""
    pytest.importorskip("torch")
    from pathbench.core.datasets.bag_dataset import BagDataset

    return BagDataset(
        f"smoke_{target_column}",
        str(workspace.feature_dir),
        str(workspace.metadata_csv),
        target_column,
    )


@pytest.mark.smoke
def test_binary_classification_mil_benchmark_grid(
    extracted_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
) -> None:
    """Run a tiny binary MIL benchmark over a small hyperparameter grid."""
    dataset = _build_bag_dataset(extracted_bag_workspace, target_column="binary_label")
    runs = []

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="hf_binary_classification_benchmark",
        metadata={"num_bags": len(dataset), "grid_size": 2},
    ) as metadata:
        for lr, dropout in ((1e-3, 0.0), (5e-4, 0.2)):
            _, result = fit_smoke_model(
                tmp_path / f"binary_{lr}_{dropout}",
                dataset_train=dataset,
                dataset_val=dataset,
                input_dim=extracted_bag_workspace.input_dim,
                output_dim=2,
                task="classification",
                loss_name="CrossEntropyLoss",
                epochs=DEFAULT_SMOKE_EPOCHS,
                lr=lr,
                dropout=dropout,
            )
            runs.append(result)
        metadata["best_scores"] = [run.best_score for run in runs]
        attach_smoke_outputs(
            metadata,
            step_name="hf_binary_classification_benchmark",
            intermediate={"bag_metadata_csv": extracted_bag_workspace.metadata_csv},
            final={
                **training_artifact_outputs(runs[0]),
                **{
                    f"run_{index}_checkpoint": Path(run.best_model_path)
                    for index, run in enumerate(runs)
                },
            },
        )

    assert len(runs) == 2
    assert all(Path(run.best_model_path).exists() for run in runs)
    assert Path(runs[0].artifacts_dir, "val_confusion_matrix.png").exists()
    assert Path(runs[0].artifacts_dir, "val_roc_auc_curve.png").exists()
    assert Path(runs[0].artifacts_dir, "val_pr_auc_curve.png").exists()
    assert Path(runs[0].artifacts_dir, "val_calibration_curve.png").exists()


@pytest.mark.smoke
def test_multiclass_classification_mil_benchmark_grid(
    extracted_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
) -> None:
    """Run a tiny multiclass MIL benchmark using the same extracted bags."""
    import pandas as pd

    dataset = _build_bag_dataset(
        extracted_bag_workspace, target_column="multiclass_label"
    )
    runs = []

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="hf_multiclass_classification_benchmark",
        metadata={"num_bags": len(dataset), "grid_size": 2},
    ) as metadata:
        for lr, dropout in ((1e-3, 0.0), (5e-4, 0.1)):
            _, result = fit_smoke_model(
                tmp_path / f"multiclass_{lr}_{dropout}",
                dataset_train=dataset,
                dataset_val=dataset,
                input_dim=extracted_bag_workspace.input_dim,
                output_dim=5,
                task="classification",
                loss_name="CrossEntropyLoss",
                epochs=DEFAULT_SMOKE_EPOCHS,
                lr=lr,
                dropout=dropout,
            )
            runs.append(result)
        attach_smoke_outputs(
            metadata,
            step_name="hf_multiclass_classification_benchmark",
            intermediate={"bag_metadata_csv": extracted_bag_workspace.metadata_csv},
            final={
                **training_artifact_outputs(runs[0]),
                **{
                    f"run_{index}_checkpoint": Path(run.best_model_path)
                    for index, run in enumerate(runs)
                },
            },
        )

    assert len(runs) == 2
    assert all(Path(run.best_model_path).exists() for run in runs)
    assert Path(runs[0].artifacts_dir, "val_confusion_matrix.png").exists()
    assert Path(runs[0].artifacts_dir, "val_roc_auc_curve.png").exists()
    assert Path(runs[0].artifacts_dir, "val_pr_auc_curve.png").exists()
    assert Path(runs[0].artifacts_dir, "val_calibration_curve.png").exists()
    metadata_df = pd.read_csv(extracted_bag_workspace.metadata_csv)
    assert metadata_df["multiclass_label"].nunique() == 5


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
        return {"logits": self.head(pooled)}, {"ignored": True}


@pytest.mark.smoke
def test_torchmil_backend_mil_benchmark(
    extracted_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Run a binary MIL benchmark using the torchmil backend (ABMIL)."""
    from pathbench.adapters.torchmil.backend import TorchMILBackendModel, register_torchmil_backend
    from pathbench.core.datasets.bag_dataset import BagDataset
    from pathbench.training.lightning import LightningTrainer
    from pathbench.utils.registries import LOSSES

    register_torchmil_backend()
    monkeypatch.setattr(
        "pathbench.adapters.torchmil.backend.require_torchmil",
        lambda feature: None,
    )
    monkeypatch.setattr(
        "pathbench.adapters.torchmil.backend.build_torchmil_model",
        lambda spec, config_kwargs: _FakeTorchMILModel(config_kwargs["in_shape"][0]),
    )

    dataset = BagDataset(
        "smoke_torchmil_binary",
        str(extracted_bag_workspace.feature_dir),
        str(extracted_bag_workspace.metadata_csv),
        "binary_label",
    )
    cfg = make_training_config(
        tmp_path / "torchmil_benchmark",
        task="classification",
        epochs=DEFAULT_SMOKE_EPOCHS,
        lr=1e-3,
        dropout=0.0,
        batch_size=1,
    )
    cfg.mil.backend = "torchmil"
    cfg.mil.torchmil_model = "ABMIL"
    cfg.mil.torchmil_model_kwargs = {
        "in_shape": (extracted_bag_workspace.input_dim,),
        "out_shape": 2,
    }
    cfg.mil.best_epoch_based_on = "balanced_accuracy"
    cfg.mil.use_torchmil_collate = False

    model = TorchMILBackendModel(
        torchmil_model="ABMIL",
        task="classification",
        torchmil_model_kwargs={
            "in_shape": (extracted_bag_workspace.input_dim,),
            "out_shape": 2,
        },
    )
    trainer = LightningTrainer(cfg)
    loss_fn = LOSSES.get("CrossEntropyLoss")()

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="hf_torchmil_mil_benchmark",
        metadata={"num_bags": len(dataset), "backend": "torchmil", "model": "ABMIL"},
    ) as metadata:
        best_model_path, best_score = trainer.fit(model, dataset, dataset, loss_fn)
        artifacts_dir = Path(cfg.experiment.project_root) / "training_artifacts"
        attach_smoke_outputs(
            metadata,
            step_name="hf_torchmil_mil_benchmark",
            intermediate={"bag_metadata_csv": extracted_bag_workspace.metadata_csv},
            final={
                "best_model_path": Path(best_model_path),
                **training_artifact_outputs(
                    SmokeTrainingResult(
                        best_model_path=str(best_model_path),
                        best_score=float(best_score),
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
def test_mil_lab_backend_mil_benchmark(
    extracted_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Run a binary MIL benchmark using the mil-lab backend (abmil)."""
    from pathbench.adapters.mil_lab.backend import MILLabBackendModel, register_mil_lab_backend
    from pathbench.core.datasets.bag_dataset import BagDataset
    from pathbench.training.lightning import LightningTrainer
    from pathbench.utils.registries import LOSSES

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

    dataset = BagDataset(
        "smoke_mil_lab_binary",
        str(extracted_bag_workspace.feature_dir),
        str(extracted_bag_workspace.metadata_csv),
        "binary_label",
    )
    cfg = make_training_config(
        tmp_path / "mil_lab_benchmark",
        task="classification",
        epochs=DEFAULT_SMOKE_EPOCHS,
        lr=1e-3,
        dropout=0.0,
        batch_size=1,
    )
    cfg.mil.backend = "mil-lab"
    cfg.mil.mil_lab_model = "abmil"
    cfg.mil.mil_lab_model_kwargs = {
        "input_dim": extracted_bag_workspace.input_dim,
        "output_dim": 2,
    }
    cfg.mil.best_epoch_based_on = "balanced_accuracy"

    model = MILLabBackendModel(
        mil_lab_model="abmil",
        task="classification",
        mil_lab_model_kwargs={
            "input_dim": extracted_bag_workspace.input_dim,
            "output_dim": 2,
        },
    )
    trainer = LightningTrainer(cfg)
    loss_fn = LOSSES.get("CrossEntropyLoss")()

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="hf_mil_lab_mil_benchmark",
        metadata={"num_bags": len(dataset), "backend": "mil-lab", "model": "abmil"},
    ) as metadata:
        best_model_path, best_score = trainer.fit(model, dataset, dataset, loss_fn)
        artifacts_dir = Path(cfg.experiment.project_root) / "training_artifacts"
        attach_smoke_outputs(
            metadata,
            step_name="hf_mil_lab_mil_benchmark",
            intermediate={"bag_metadata_csv": extracted_bag_workspace.metadata_csv},
            final={
                "best_model_path": Path(best_model_path),
                **training_artifact_outputs(
                    SmokeTrainingResult(
                        best_model_path=str(best_model_path),
                        best_score=float(best_score),
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
def test_heatmap_overlays_from_benchmark_models(
    extracted_bag_workspace: PreparedBagWorkspace,
    extracted_wsi_workspace: ExtractedWsiWorkspace,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Generate tile-level heatmap overlays for all three MIL benchmark backends.

    Uses real VarMIL attention scores for the lightning/native backend and
    deterministic synthetic scores for the torchmil and mil-lab backends (whose
    smoke models are mean-pool fakes without per-instance outputs).  All three
    heatmaps are written to the slide H5 artifact and exported as PNG overlays.
    """
    from pathbench.adapters.torchmil.heatmap_explainer import register_torchmil_heatmap_explainer
    from pathbench.core.datasets.bag_dataset import BagDataset
    from pathbench.core.io.h5 import heatmaps as heatmap_io
    from pathbench.core.io.h5.base import FileHandleH5
    from pathbench.inference.heatmaps import create_inference_heatmap

    register_torchmil_heatmap_explainer()
    monkeypatch.setattr(
        "pathbench.inference.heatmaps.require_torchmil",
        lambda feature: None,
    )
    monkeypatch.setattr(
        "pathbench.adapters.torchmil.heatmap_explainer.require_torchmil",
        lambda feature: None,
    )

    dataset = BagDataset(
        "smoke_heatmap",
        str(extracted_bag_workspace.feature_dir),
        str(extracted_bag_workspace.metadata_csv),
        "binary_label",
    )
    native_model, _result = fit_smoke_model(
        tmp_path / "heatmap_train",
        dataset_train=dataset,
        dataset_val=dataset,
        input_dim=extracted_bag_workspace.input_dim,
        output_dim=2,
        task="classification",
        loss_name="CrossEntropyLoss",
        epochs=DEFAULT_SMOKE_EPOCHS,
        lr=1e-3,
        dropout=0.0,
    )

    slide_id = extracted_bag_workspace.slide_ids[0]
    artifact_path = extracted_wsi_workspace.artifact_paths[slide_id]
    bag_path = extracted_bag_workspace.feature_dir / f"{slide_id}.pt"
    bag_tensor = torch.load(bag_path, weights_only=True).unsqueeze(0)
    num_tiles = int(bag_tensor.shape[1])

    attention_scores = (
        native_model.instance_scores(bag_tensor)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    if attention_scores.ndim > 1:
        attention_scores = attention_scores.squeeze(-1)

    rng = np.random.default_rng(42)
    backends: dict[str, np.ndarray] = {
        "lightning": attention_scores,
        "torchmil": rng.random(num_tiles).astype(np.float32),
        "mil_lab": rng.random(num_tiles).astype(np.float32),
    }

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="hf_heatmap_overlays",
        metadata={
            "slide_id": slide_id,
            "num_tiles": num_tiles,
            "num_backends": len(backends),
        },
    ) as metadata:
        final_outputs: dict[str, Path] = {}
        for backend_name, scores in backends.items():
            scores_path = tmp_path / f"{backend_name}_scores.npy"
            heatmap_png = tmp_path / f"{backend_name}_heatmap.png"
            np.save(scores_path, scores)
            create_inference_heatmap(
                artifact_path=artifact_path,
                bag_id=extracted_wsi_workspace.bag_id,
                scores_path=scores_path,
                heatmap_backend="torchmil",
                heatmap_name=f"smoke_{backend_name}_attention",
                image_output_path=heatmap_png,
                slide_path=extracted_wsi_workspace.slides_dir / f"{slide_id}.svs",
            )
            final_outputs[f"{backend_name}_heatmap_png"] = heatmap_png

        attach_smoke_outputs(
            metadata,
            step_name="hf_heatmap_overlays",
            intermediate={"artifact_path": artifact_path},
            final=final_outputs,
        )

    for backend_name in backends:
        heatmap_png = tmp_path / f"{backend_name}_heatmap.png"
        assert heatmap_png.exists(), f"Heatmap PNG missing for backend '{backend_name}'"
        with FileHandleH5(artifact_path, mode="r") as slide_artifact:
            heatmap = heatmap_io.read_prediction_heatmap(
                slide_artifact,
                extracted_wsi_workspace.bag_id,
                f"smoke_{backend_name}_attention",
            )
        assert heatmap["coords"].shape[0] == num_tiles
        assert heatmap["scores"].shape == (num_tiles,)
        assert np.isfinite(heatmap["scores"]).all()
        assert float(heatmap["scores"].min()) >= 0.0
        assert float(heatmap["scores"].max()) <= 1.0
