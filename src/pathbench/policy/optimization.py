from __future__ import annotations
"""Optimization policy that samples the configuration space using Optuna."""

from pathlib import Path
from typing import Any, Dict, Sequence, Mapping
import inspect

import optuna

from pathbench.config.config import Config
from pathbench.core.datasets.bag_dataset import BagDataset
from pathbench.core.experiments.base import ComboConfig, Experiment
from pathbench.core.losses.base import Loss
from pathbench.core.models.mil_base import MILModelBase
from pathbench.policy.base import PolicyBase
from pathbench.policy.feature_extraction import FeatureExtractionPolicy
from pathbench.utils.splitting import build_patient_splits
from pathbench.utils.registries import LOSSES, MODELS, TRAINERS

class OptimizationPolicy(PolicyBase):
   """Optuna-driven search over the MIL configuration space."""

    def __init__(self, experiment: Experiment) -> None:
        super().__init__(experiment)
        self.config: Config = experiment.cfg
        self.datasets = experiment.build_datasets()
        self.annotations = experiment.load_annotations()
        self.feature_policy = FeatureExtractionPolicy(experiment)
        self.project_root = Path(experiment.project_root or ".")
        self.best_trial: optuna.trial.FrozenTrial | None = None
        self.best_model: MILModelBase | None = None
        self.best_combo: ComboConfig | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    def execute(self) -> Dict[str, Any]:
        self.feature_policy.execute()
        
        sampler = self._get_sampler()
        pruner = self._get_pruner()
        
        direction = "maximize" if self.config.optimization.objective_mode == "max" else "minimize"
        
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            study_name=self.config.optimization.study_name,
            load_if_exists=self.config.optimization.load_study,
        )

        study.optimize(self._objective, n_trials=self.config.optimization.trials)

        self.best_trial = study.best_trial
        results_path = self.project_root / f"{self.config.optimization.study_name}_results.csv"
        study.trials_dataframe().to_csv(results_path, index=False)

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "best_model": self.best_model,
            "best_configuration": self.best_combo.to_dict() if self.best_combo else None,
            "study": study,
        }
        
    # ------------------------------------------------------------------
    # Objective + helpers
    # ------------------------------------------------------------------
    def _objective(self, trial: optuna.Trial) -> float:
        combo = self._sample_combo(trial)
        model_cls = MODELS.get(combo.mil)
        loss_cls = LOSSES.get(combo.loss)
        trainer_cls = TRAINERS.get("lightning")

        if model_cls is None or loss_cls is None or trainer_cls is None:
            raise RuntimeError("Missing model, loss, or trainer registration for optimization.")

        annotation_splits = build_patient_splits(self.annotations, self.config, self.datasets)

        train_ds = self._build_split_dataset("training", annotation_splits)
        val_ds = self._build_split_dataset("validation", annotation_splits)

        model: MILModelBase = model_cls(**self._model_kwargs(model_cls))
        loss_fn: Loss = loss_cls()
        trainer = trainer_cls(self.config)

        best_path, best_score = trainer.fit(model, train_ds, val_ds, loss_fn)
        metric_value = float(best_score)

        # Track best overall
        if self.best_trial is None or self._is_better(metric_value, self.best_trial.value):
            self.best_trial = trial.freeze()
            self.best_model = model
            self.best_combo = combo

        return metric_value

    def _sample_combo(self, trial: optuna.Trial) -> ComboConfig:
        bp = self.config.benchmark_parameters.model_dump()

        sampled: Dict[str, Any] = {}
        for key, values in bp.items():
            if isinstance(values, Sequence) and values:
                sampled[key] = trial.suggest_categorical(key, values)

        # fallbacks
        sampled.setdefault("mil", "AttentionMIL")
        sampled.setdefault("loss", "CrossEntropyLoss")

        return ComboConfig(**sampled)

    def _build_split_dataset(self, usage: str, splits: Mapping[str, pd.DataFrame]) -> BagDataset:
        feature_dir = self.project_root / "features"
        coord_dir = (self.project_root / "tile_coords") if (self.project_root / "tile_coords").exists() else None

        return BagDataset(
            name=usage,
            annotations=splits[usage],
            feature_dir=feature_dir,
            coord_dir=coord_dir,
            label_column=self.config.experiment.label_column,
            bag_level=self.config.experiment.aggregation_level,
            patient_column=self.config.experiment.patient_column,
        )

    def _model_kwargs(self, model_cls: type) -> Dict[str, Any]:
        mil_options = self.config.mil.model_dump()
        signature = inspect.signature(model_cls)
        return {k: v for k, v in mil_options.items() if k in signature.parameters}

    def _get_sampler(self) -> optuna.samplers.BaseSampler:
        name = self.config.optimization.sampler
        seed = 42
        if name == "TPESampler":
            return optuna.samplers.TPESampler(seed=seed)
        if name == "RandomSampler":
            return optuna.samplers.RandomSampler(seed=seed)
        if name == "CmaEsSampler":
            return optuna.samplers.CmaEsSampler(seed=seed)
        return optuna.samplers.TPESampler(seed=seed)

    def _get_pruner(self) -> optuna.pruners.BasePruner:
        name = self.config.optimization.pruner or "NopPruner"
        if name == "MedianPruner":
            return optuna.pruners.MedianPruner()
        if name == "HyperbandPruner":
            return optuna.pruners.HyperbandPruner()
        return optuna.pruners.NopPruner()

    def _is_better(self, candidate: float, best: float) -> bool:
        if self.config.optimization.objective_mode == "min":
            return candidate < best
        return candidate > best