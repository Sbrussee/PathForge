"""Trainer for scikit-learn / scikit-survival slide-level estimators.

Usage::

    from pathbench.core.models.sklearn_slide import SklearnLogisticRegressionClassifier
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    model = SklearnLogisticRegressionClassifier()
    trainer = SklearnSlideTrainer(output_dir=Path("runs/lr"), task="classification")
    model_path, score = trainer.fit(model, X_train, y_train, X_val, y_val)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from pathbench.core.models.base import ScikitBase
from pathbench.training.base import TrainerBase
from pathbench.training.metrics import (
    compute_task_metrics,
    save_task_evaluation_artifacts,
)


class SklearnSlideTrainer(TrainerBase):
    """Fit a :class:`~pathbench.core.models.base.ScikitBase` estimator on
    slide-level numpy feature arrays and persist evaluation artefacts.

    This trainer does **not** use PyTorch Lightning; it calls sklearn's
    ``fit``/``predict`` directly and then runs the same
    :func:`~pathbench.training.metrics.save_task_evaluation_artifacts` pipeline
    used by :class:`~pathbench.training.lightning.LightningTrainer`.

    Args:
        output_dir: Root directory; model pickle and artifacts go here.
        task: PathBench task name — one of ``"classification"``,
            ``"regression"``, ``"survival"``, ``"survival_discrete"``.
        selected_metrics: Optional list of metric names to compute; defaults
            to all metrics for the task.
    """

    def __init__(
        self,
        output_dir: Path | str,
        task: str,
        selected_metrics: list[str] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.task = task
        self.selected_metrics = selected_metrics
        self.metrics_artifacts_dir = self.output_dir / "training_artifacts"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        model: ScikitBase,
        X_train: np.ndarray,
        y_train: Any,
        X_val: np.ndarray,
        y_val: Any,
    ) -> tuple[str, float]:
        """Fit *model* and evaluate on the validation set.

        Args:
            model: Unfitted sklearn estimator wrapper.
            X_train: Training feature matrix shaped ``[N_train, D]``.
            y_train: Training targets.  For classification/regression: a 1-D
                numpy array; for survival: a dict with ``"time"`` and
                ``"event"`` keys (1-D numpy arrays).
            X_val: Validation feature matrix shaped ``[N_val, D]``.
            y_val: Validation targets with the same structure as *y_train*.

        Returns:
            Tuple of ``(model_pickle_path, best_scalar_score)`` where the
            score is the primary validation metric for the task
            (``balanced_accuracy`` for classification, ``c_index`` for
            survival, ``r2`` for regression).
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_artifacts_dir.mkdir(parents=True, exist_ok=True)

        # --- fit -----------------------------------------------------------
        if self.task in {"survival", "survival_discrete"}:
            from pathbench.core.models.sklearn_slide import (
                SklearnSlideClassifier,
                make_survival_structured_array,
            )

            if isinstance(model, SklearnSlideClassifier):
                # Discrete survival as multiclass classification: fit on bin indices.
                # SklearnSlideClassifier.predict_as_tensor returns log-probabilities
                # (N, T) which are compatible with survival_discrete metrics.
                time = y_train["time"] if isinstance(y_train, dict) else y_train
                model.fit(X_train, np.asarray(time, dtype=np.int64))
            else:
                # Continuous or sksurv survival estimator: requires structured array.
                time = y_train["time"] if isinstance(y_train, dict) else y_train
                event = (
                    y_train["event"] if isinstance(y_train, dict) else np.ones_like(time)
                )
                y_train_structured = make_survival_structured_array(
                    np.asarray(time, dtype=np.float64),
                    np.asarray(event, dtype=np.float64),
                )
                model.fit(X_train, y_train_structured)
        else:
            model.fit(X_train, np.asarray(y_train))

        # --- predict on val ------------------------------------------------
        predictions = model.predict_as_tensor(X_val)
        target = self._build_target_tensor(y_val)

        # --- save artefacts ------------------------------------------------
        save_task_evaluation_artifacts(
            predictions,
            target,
            task=self.task,
            output_dir=self.metrics_artifacts_dir,
            prefix="val",
            selected_metrics=self.selected_metrics,
        )

        # --- persist model -------------------------------------------------
        model_path = self.output_dir / "sklearn_model.pkl"
        model.save(str(model_path))

        # --- compute primary score -----------------------------------------
        metrics = compute_task_metrics(predictions, target, task=self.task)
        primary_key = _primary_metric(self.task)
        best_score = float(metrics.get(primary_key, 0.0))

        return str(model_path), best_score

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_target_tensor(self, y: Any) -> Any:
        if self.task in {"survival", "survival_discrete"}:
            if isinstance(y, dict):
                return {
                    "time": torch.from_numpy(np.asarray(y["time"], dtype=np.float32)),
                    "event": torch.from_numpy(np.asarray(y["event"], dtype=np.float32)),
                }
            raise ValueError(
                "Survival y_val must be a dict with 'time' and 'event' keys."
            )
        if self.task == "classification":
            return torch.from_numpy(np.asarray(y, dtype=np.int64))
        if self.task == "regression":
            return torch.from_numpy(np.asarray(y, dtype=np.float32))
        raise ValueError(f"Unknown task: {self.task!r}")

    def predict(self, model: ScikitBase, dataset: np.ndarray) -> torch.Tensor:
        """Run prediction for a slide-level feature matrix shaped ``[N, D]``."""
        return model.predict_as_tensor(dataset)


def _primary_metric(task: str) -> str:
    return {
        "classification": "balanced_accuracy",
        "survival": "c_index",
        "survival_discrete": "c_index",
        "regression": "r2",
    }.get(task, "r2")
