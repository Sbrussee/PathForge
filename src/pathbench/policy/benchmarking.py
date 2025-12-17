from __future__ import annotations
"""Benchmarking policy for exhaustive evaluation of configuration grids."""

from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from pathbench.config.config import Config
from pathbench.core.datasets.bag_dataset import BagDataset
from pathbench.core.experiments.base import ComboConfig, Experiment
from pathbench.core.losses.base import Loss
from pathbench.core.models.mil_base import MILModelBase
from pathbench.policy.base import PolicyBase
from pathbench.policy.feature_extraction import FeatureExtractionPolicy
from pathbench.utils.splitting import build_patient_splits
from pathbench.training.base import TrainerBase
from pathbench.utils.registries import LOSSES, MODELS, TRAINERS


class BenchmarkingPolicy(PolicyBase):
    """Run benchmark experiments over every configuration combination.

    The policy wires together the entire MIL pipeline:
    1. Run tissue detection, tiling and feature extraction (delegated to
       :class:`FeatureExtractionPolicy`).
    2. Build bag datasets at slide, patient or tissue level.
    3. Train and optionally cross-validate the MIL model.
    4. Evaluate and persist a results matrix covering train/val/test splits for
       each configuration and fold.
    """


    def __init__(self, experiment: Experiment) -> None:
        super().__init__(experiment)
        self.config: Config = experiment.cfg
        self.feature_policy = FeatureExtractionPolicy(experiment)
        self.datasets = experiment.build_datasets()
        self.combos = self._build_combo_grid()
        self.annotations = experiment.load_annotations()
        self.project_root = Path(experiment.project_root or ".")
        self.results: List[Dict[str, Any]] = []
        self.experiment = experiment
        self.config = config

        # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def execute(self, cv_folds: int | None = None) -> List[Dict[str, Any]]:
        """Execute the benchmarking run.

        Parameters
        ----------
        cv_folds:
            Number of cross-validation folds. When ``None`` the value is derived
            from ``config.experiment.split_technique`` (defaults to ``1``).
        """

        self.feature_policy.execute()
        annotation_splits = build_patient_splits(self.annotations, self.config, self.datasets)

        train_ds = self._build_split_dataset("training", annotation_splits)
        val_ds = self._build_split_dataset("validation", annotation_splits)
        test_ds = self._build_split_dataset("testing", annotation_splits)

        folds = cv_folds or self._infer_folds()

        for combo_idx, combo in enumerate(self.combos):
            fold_results = self._run_configuration(combo, train_ds, val_ds, test_ds, folds)
            self.results.extend(fold_results)

        self._save_results()
        return self.results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_combo_grid(self) -> List[ComboConfig]:
        bp_dict = self.config.benchmark_parameters.model_dump()
        keys = [k for k, v in bp_dict.items() if isinstance(v, Sequence) and len(v) > 0]
        values = [bp_dict[k] for k in keys]
        combos: List[ComboConfig] = []
        for vals in product(*values):
            combos.append(ComboConfig.from_keys_values(keys, list(vals)))
        return combos

    def _infer_folds(self) -> int:
        technique = self.config.experiment.split_technique
        if technique.startswith("k-fold"):
            return getattr(self.config.experiment, "cv_folds", 5)
        return 1

    def _build_split_dataset(self, usage: str, splits: Mapping[str, pd.DataFrame]) -> BagDataset:
        feature_dir = self._feature_root()
        coord_dir = self._coordinate_root()

        return BagDataset(
            name=usage,
            annotations=splits[usage],
            feature_dir=feature_dir,
            coord_dir=coord_dir,
            label_column=self.config.experiment.label_column,
            bag_level=self.config.experiment.aggregation_level,
            patient_column=self.config.experiment.patient_column,
        )

    def _feature_root(self) -> Path:
        candidate = self.project_root / "features"
        return candidate if candidate.exists() else Path(self.datasets[0].features_dir)

    def _coordinate_root(self) -> Path | None:
        candidate = self.project_root / "tile_coords"
        return candidate if candidate.exists() else None

    def _run_configuration(
        self,
        combo: ComboConfig,
        train_ds: BagDataset,
        val_ds: BagDataset,
        test_ds: BagDataset,
        folds: int,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        trainer = self._resolve_trainer()
        model_cls = MODELS.get(getattr(combo, "mil", None) or getattr(combo, "model", None))
        loss_cls = LOSSES.get(getattr(combo, "loss", "CrossEntropyLoss"))

        if model_cls is None or loss_cls is None:
            raise ValueError("Model or loss configuration missing in combo.")

        model: MILModelBase = model_cls(**self._model_kwargs(model_cls))
        loss_fn: Loss = loss_cls()

        if folds <= 1:
            fold_result = self._train_evaluate(trainer, model, loss_fn, train_ds, val_ds, test_ds, combo, fold_idx=0)
            results.append(fold_result)
            return results

        patient_groups = train_ds.group_indices_by_patient()
        patient_ids = list(patient_groups.keys())
        np.random.shuffle(patient_ids)
        fold_patients = np.array_split(patient_ids, folds)

        for fold_idx in range(folds):
            val_patients = set(fold_patients[fold_idx])
            train_patients = [p for i, arr in enumerate(fold_patients) if i != fold_idx for p in arr]

            val_indices = [idx for p in val_patients for idx in patient_groups[p]]
            train_indices = [idx for p in train_patients for idx in patient_groups[p]]

            fold_train = Subset(train_ds, train_indices)
            fold_val = Subset(train_ds, val_indices)

            fold_result = self._train_evaluate(
                trainer,
                model_cls(**self._model_kwargs(model_cls)),
                loss_cls(),
                fold_train,
                fold_val,
                test_ds,
                combo,
                fold_idx,
            )
            results.append(fold_result)

        return results

    def _train_evaluate(
        self,
        trainer: TrainerBase,
        model: MILModelBase,
        loss_fn: Loss,
        train_ds: Iterable,
        val_ds: Iterable,
        test_ds: Iterable,
        combo: ComboConfig,
        fold_idx: int,
    ) -> Dict[str, Any]:
        best_path, best_score = trainer.fit(model, train_ds, val_ds, loss_fn)

        train_metrics = self._evaluate(model, train_ds, loss_fn)
        val_metrics = self._evaluate(model, val_ds, loss_fn)
        test_metrics = self._evaluate(model, test_ds, loss_fn)

        return {
            "combo": combo.to_dict(),
            "fold": fold_idx,
            "best_checkpoint": best_path,
            "best_score": float(best_score),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        
  def _resolve_trainer(self) -> TrainerBase:
        trainer_backend = "lightning"
        trainer_cls = TRAINERS.get(trainer_backend)
        if trainer_cls is None:
            raise ValueError(f"Trainer backend '{trainer_backend}' not found.")
        return trainer_cls(self.config)

    def _model_kwargs(self, model_cls: type) -> Dict[str, Any]:
        """Filter MIL config options for model construction."""

        mil_options = self.config.mil.model_dump()
        signature = inspect.signature(model_cls)
        allowed = {k: v for k, v in mil_options.items() if k in signature.parameters}
        return allowed

    def _evaluate(self, model: MILModelBase, dataset: Iterable, loss_fn: Loss) -> Mapping[str, float]:
        loader = DataLoader(dataset, batch_size=self.config.mil.batch_size, shuffle=False)
        model.eval()
        losses: List[float] = []
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                bag, target = batch
                logits = model(bag["features"] if isinstance(bag, dict) else bag)
                loss_val = loss_fn(logits, target)
                losses.append(loss_val.item())

                preds = torch.argmax(logits, dim=1)
                correct += (preds == target).sum().item()
                total += target.numel()

        return {
            "loss": float(np.mean(losses) if losses else 0.0),
            "accuracy": float(correct / total) if total > 0 else 0.0,
        }

    def _save_results(self) -> None:
        df = pd.DataFrame(self.results)
        out_path = self.project_root / "benchmark_results.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)