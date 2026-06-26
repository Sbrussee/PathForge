"""Reusable training primitives for the smoke-test suite.

The production benchmarking and optimization policies are still evolving, so
the smoke suite exercises the stable registry, dataset, trainer, and inference
contracts directly with tiny deterministic models and losses.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SMOKE_EPOCHS = int(os.environ.get("PATHFORGE_SMOKE_EPOCHS", "5"))
DEFAULT_SMOKE_BATCH_SIZE = int(os.environ.get("PATHFORGE_SMOKE_BATCH_SIZE", "2"))


@dataclass(frozen=True)
class SmokeTrainingResult:
    """Summary of one tiny MIL training run.

    Attributes:
        best_model_path: Best checkpoint returned by ``LightningTrainer.fit``.
        best_score: Scalar validation score selected by the checkpoint callback.
        output_dim: Number of output channels produced by the trained model.
        task_name: PathForge task name used for the run.
        artifacts_dir: Directory containing post-training metric JSON and plots.
        config: Optional PathForge Config used during training.
    """

    best_model_path: str
    best_score: float
    output_dim: int
    task_name: str
    artifacts_dir: str
    config: Any | None = None


def training_artifact_outputs(result: SmokeTrainingResult) -> dict[str, Path]:
    """Return explicit training artifact paths for smoke-report attachment.

    Args:
        result: Completed smoke-training result.

    Returns:
        dict[str, Path]: Named metric files and plots. Classification tasks
        expose ROC and confusion-matrix plots; survival tasks expose TD-AUC
        and concordance plots.
    """

    artifacts_dir = Path(result.artifacts_dir)
    outputs = {
        "val_metrics_json": artifacts_dir / "val_metrics.json",
        "val_curves_json": artifacts_dir / "val_curves.json",
    }
    if result.task_name == "classification":
        outputs["val_confusion_matrix_png"] = artifacts_dir / "val_confusion_matrix.png"
        outputs["val_roc_auc_curve_png"] = artifacts_dir / "val_roc_auc_curve.png"
        outputs["val_pr_auc_curve_png"] = artifacts_dir / "val_pr_auc_curve.png"
        outputs["val_calibration_curve_png"] = artifacts_dir / "val_calibration_curve.png"
    elif result.task_name in {"survival", "survival_discrete"}:
        outputs["val_td_auc_curve_png"] = artifacts_dir / "val_td_auc_curve.png"
        outputs["val_concordance_index_png"] = artifacts_dir / "val_concordance_index.png"
        outputs["val_kaplan_meier_png"] = artifacts_dir / "val_kaplan_meier.png"
    return outputs


class SurvivalBagDataset:
    """Small bag dataset for survival smoke tests.

    Args:
        metadata_df: DataFrame containing ``slide_id``, one time column, and one
            event column.
        feature_dir: Directory containing one ``{slide_id}.pt`` tensor per row.
        time_column: Survival time column name.
        event_column: Event indicator column name.
        discrete_time: When true, return integer time bins for discrete
            survival; otherwise return floating survival times.

    Returns from ``__getitem__``:
        BagBatch dict with ``X`` shaped ``[num_instances, feature_dim]`` and ``Y``
        as a target dict with ``time`` and ``event`` tensors shaped ``[]``.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        *,
        feature_dir: Path,
        time_column: str,
        event_column: str,
        discrete_time: bool,
    ) -> None:
        import torch

        self._torch = torch
        self.metadata_df = metadata_df.reset_index(drop=True).copy()
        self.feature_dir = feature_dir
        self.time_column = time_column
        self.event_column = event_column
        self.discrete_time = discrete_time

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.metadata_df.iloc[index]
        bag = self._torch.load(self.feature_dir / f"{row['slide_id']}.pt")
        time_value = row[self.time_column]
        event_value = row[self.event_column]
        target = {
            "time": self._torch.tensor(
                int(time_value) if self.discrete_time else float(time_value),
                dtype=self._torch.long if self.discrete_time else self._torch.float32,
            ),
            "event": self._torch.tensor(float(event_value), dtype=self._torch.float32),
        }
        if self.discrete_time and "os_months" in row.index:
            target["continuous_time"] = self._torch.tensor(
                float(row["os_months"]),
                dtype=self._torch.float32,
            )
        return {"X": bag.float(), "Y": target}


def register_smoke_components() -> None:
    """Register production components required by the smoke suite once."""

    from pathforge.utils.registries import populate_dynamic_registries

    populate_dynamic_registries()


def _best_epoch_monitor(task: str) -> str:
    if task == "classification":
        return "balanced_accuracy"
    if task in {"survival", "survival_discrete"}:
        return "c_index"
    return "val_loss"


def _build_real_smoke_model(
    *,
    input_dim: int,
    output_dim: int,
) -> Any:
    """Instantiate a lightweight production MIL model for smoke training.

    Args:
        input_dim: Feature dimension ``D`` per instance.
        output_dim: Number of task outputs.

    Returns:
        Any: Configured production MIL model instance.
    """

    from pathforge.utils.registries import MODELS

    ModelClass = MODELS.get("VarMIL")
    return ModelClass(input_dim=input_dim, hidden_dim=64, output_dim=output_dim)


def _smoke_batch_size(task: str, dataset: Any) -> int:
    """Choose a stable batch size for real smoke training.

    Native PathForge MIL training now reuses the padded collate adapter when
    ``batch_size > 1``. Classification smoke runs can therefore use a small
    multi-bag batch, while survival smoke still benefits from a full-batch
    Cox/discrete-survival objective.
    """

    if task == "classification":
        return min(DEFAULT_SMOKE_BATCH_SIZE, max(1, len(dataset)))
    return max(DEFAULT_SMOKE_BATCH_SIZE, len(dataset))


def make_training_config(
    root_dir: Path,
    *,
    task: str,
    epochs: int,
    lr: float,
    dropout: float,
    batch_size: int = 1,
) -> Any:
    """Create a minimal writable PathForge config for trainer smoke tests."""
    from pathforge.config.config import Config

    root_dir.mkdir(parents=True, exist_ok=True)
    annotations_csv = root_dir / "annotations.csv"
    annotations_csv.write_text("dataset,slide,patient,category\n", encoding="utf-8")
    slides_dir = root_dir / "slides"
    artifacts_dir = root_dir / "artifacts"
    slides_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config.model_validate(
        {
            "experiment": {
                "project_name": f"smoke_{task}",
                "annotation_file": str(annotations_csv),
                "project_root": str((root_dir / "project").resolve()),
                "mode": "benchmark",
                "task": task,
                "num_workers": 0,
            },
            "mil": {
                "backend": "native",
                "epochs": epochs,
                "lr": lr,
                "dropout_p": dropout,
                "batch_size": batch_size,
                "patience": 2,
                "best_epoch_based_on": _best_epoch_monitor(task),
            },
            "metrics": {
                "classification_backend": "native",
                "survival_continuous_backend": "native",
            },
            "slide_processing": {"backend": "lazyslide"},
            "datasets": [
                {
                    "name": "smoke_dataset",
                    "slides_dir": str(slides_dir.resolve()),
                    "artifacts_dir": str(artifacts_dir.resolve()),
                    "used_for": "all",
                }
            ],
            "benchmark_parameters": {
                "tile_px": [224],
                "tile_mpp": [1.0],
                "feature_extraction": ["resnet18"],
                "mil": ["VarMIL"],
                "loss": ["CrossEntropyLoss"],
            },
        }
    )
    return cfg


def fit_smoke_model(
    root_dir: Path,
    *,
    dataset_train: Any,
    dataset_val: Any,
    input_dim: int,
    output_dim: int,
    task: str,
    loss_name: str,
    epochs: int,
    lr: float,
    dropout: float,
) -> tuple[Any, SmokeTrainingResult]:
    """Train one tiny MIL model through the production Lightning trainer."""
    import torch

    from pathforge.training.lightning import LightningTrainer
    from pathforge.utils.registries import LOSSES

    register_smoke_components()
    batch_size = _smoke_batch_size(task, dataset_train)
    cfg = make_training_config(
        root_dir,
        task=task,
        epochs=epochs,
        lr=lr,
        dropout=dropout,
        batch_size=batch_size,
    )
    torch.manual_seed(7)
    model = _build_real_smoke_model(input_dim=input_dim, output_dim=output_dim)
    loss_fn = LOSSES.get(loss_name)()
    trainer = LightningTrainer(cfg)
    best_model_path, best_score = trainer.fit(
        model, dataset_train, dataset_val, loss_fn
    )
    return model, SmokeTrainingResult(
        best_model_path=best_model_path,
        best_score=float(best_score),
        output_dim=output_dim,
        task_name=task,
        artifacts_dir=str(trainer.metrics_artifacts_dir),
        config=cfg,
    )
