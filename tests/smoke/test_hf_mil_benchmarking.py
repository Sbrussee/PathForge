"""Smoke benchmarking workflows backed by reusable extracted bags."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ._smoke_dataset import (
    ExtractedWsiWorkspace,
    PreparedBagWorkspace,
    attach_smoke_outputs,
    capture_smoke_metrics,
)
from ._smoke_training import DEFAULT_SMOKE_EPOCHS, fit_smoke_model
from ._smoke_training import make_training_config


def _build_bag_dataset(workspace: PreparedBagWorkspace, *, target_column: str):
    """Construct a production ``BagDataset`` for one target column."""
    pytest.importorskip("torch")
    from pathforge.core.datasets.bag_dataset import BagDataset

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
                **{
                    f"run_{index}_checkpoint": Path(run.best_model_path)
                    for index, run in enumerate(runs)
                },
            },
        )

    from pathforge.policy.utils import (
        collect_run_summary_row,
        metric_should_minimize,
        save_benchmark_visualizations,
        write_experiment_summary_csv,
    )

    objective_metric = "balanced_accuracy"
    minimize = metric_should_minimize(objective_metric)
    summary_rows = [
        collect_run_summary_row(
            run.config,
            run_index=idx,
            status="success",
            objective_metric=objective_metric,
            objective_value=run.best_score,
            checkpoint_path=run.best_model_path,
        )
        for idx, run in enumerate(runs)
    ]
    agg_csv = tmp_path / "benchmark_results.csv"
    vis_dir = tmp_path / "benchmark_visualizations"
    with capture_smoke_metrics(
        tmp_path / "agg_metrics",
        step_name="smoke_benchmark_policy_aggregate",
        metadata={"grid_size": len(runs), "task": "classification"},
    ) as agg_meta:
        write_experiment_summary_csv(
            summary_rows,
            output_path=agg_csv,
            objective_metric=objective_metric,
            minimize=minimize,
        )
        save_benchmark_visualizations(
            agg_csv,
            output_dir=vis_dir,
            objective_metric=objective_metric,
            minimize=minimize,
        )
        attach_smoke_outputs(
            agg_meta,
            step_name="smoke_benchmark_policy_aggregate",
            final={
                "benchmark_results_csv": agg_csv,
                "benchmark_performance_ranked_html": vis_dir / "benchmark_performance_ranked.html",
                "benchmark_rank_scatter_html": vis_dir / "benchmark_rank_scatter.html",
            },
        )

    assert len(runs) == 2
    assert all(Path(run.best_model_path).exists() for run in runs)
    assert Path(runs[0].artifacts_dir, "val_confusion_matrix.png").exists()
    assert Path(runs[0].artifacts_dir, "val_roc_auc_curve.png").exists()
    assert Path(runs[0].artifacts_dir, "val_pr_auc_curve.png").exists()
    assert Path(runs[0].artifacts_dir, "val_calibration_curve.png").exists()

    df = pd.read_csv(agg_csv)
    assert {"run_index", "status", "objective_metric", "objective_value", "rank"}.issubset(df.columns)
    assert len(df) == 2
    assert df["objective_value"].dropna().is_monotonic_decreasing
    successful = df[df["status"] == "success"]
    assert successful["rank"].dropna().tolist() == list(range(1, len(successful) + 1))


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


@pytest.mark.smoke
def test_torchmil_backend_mil_benchmark(
    extracted_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
) -> None:
    """Run a binary MIL benchmark using the torchmil backend (ABMIL)."""
    from pathforge.adapters.torchmil.backend import TorchMILBackendModel, register_torchmil_backend
    from pathforge.core.datasets.bag_dataset import BagDataset
    from pathforge.training.lightning import LightningTrainer
    from pathforge.utils.registries import LOSSES

    register_torchmil_backend()

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
    cfg.mil.torchmil_model_kwargs = {"in_shape": (extracted_bag_workspace.input_dim,)}
    cfg.mil.best_epoch_based_on = "balanced_accuracy"
    cfg.mil.use_torchmil_collate = False

    model = TorchMILBackendModel(
        torchmil_model="ABMIL",
        task="classification",
        torchmil_model_kwargs={"in_shape": (extracted_bag_workspace.input_dim,)},
    )
    trainer = LightningTrainer(cfg)
    loss_fn = LOSSES.get("CrossEntropyLoss")()

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="hf_torchmil_mil_benchmark",
        metadata={"num_bags": len(dataset), "backend": "torchmil", "model": "ABMIL"},
    ) as metadata:
        best_model_path, _ = trainer.fit(model, dataset, dataset, loss_fn)
        artifacts_dir = Path(cfg.experiment.project_root) / "training_artifacts"
        attach_smoke_outputs(
            metadata,
            step_name="hf_torchmil_mil_benchmark",
            intermediate={"bag_metadata_csv": extracted_bag_workspace.metadata_csv},
            final={"best_model_path": Path(best_model_path)},
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
) -> None:
    """Run a binary MIL benchmark using the mil-lab backend (abmil)."""
    pytest.importorskip("mil_lab")
    from pathforge.adapters.mil_lab.backend import MILLabBackendModel, register_mil_lab_backend
    from pathforge.core.datasets.bag_dataset import BagDataset
    from pathforge.training.lightning import LightningTrainer
    from pathforge.utils.registries import LOSSES

    register_mil_lab_backend()

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
        best_model_path, _ = trainer.fit(model, dataset, dataset, loss_fn)
        artifacts_dir = Path(cfg.experiment.project_root) / "training_artifacts"
        attach_smoke_outputs(
            metadata,
            step_name="hf_mil_lab_mil_benchmark",
            intermediate={"bag_metadata_csv": extracted_bag_workspace.metadata_csv},
            final={"best_model_path": Path(best_model_path)},
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
) -> None:
    """Generate tile-level heatmap overlays for all three MIL benchmark backends.

    Uses real VarMIL attention scores for the lightning/native backend and
    random scores for the torchmil and mil-lab backends to cover the heatmap
    rendering path.  All heatmaps are written to the slide H5 artifact and
    exported as PNG overlays.
    """
    from pathforge.adapters.torchmil.heatmap_explainer import register_torchmil_heatmap_explainer
    from pathforge.core.datasets.bag_dataset import BagDataset
    from pathforge.core.io.h5 import heatmaps as heatmap_io
    from pathforge.core.io.h5.base import FileHandleH5
    from pathforge.inference.heatmaps import create_inference_heatmap

    register_torchmil_heatmap_explainer()

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
