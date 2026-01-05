from __future__ import annotations
from typing import Any, Dict, Optional
import optuna
import pandas as pd
from pathbench.policy.base import PolicyBase
from pathbench.config.config import Config
from pathbench.utils.registries import MODELS, LOSSES, TRAINERS
from pathbench.core.datasets.bag_dataset import BagDataset
from pathbench.training.base import TrainerBase

class OptimizationPolicy(PolicyBase):
    """
    Optimization Policy that dynamically configures Optuna based on Config.
    """
    
    def __init__(self, config: Config) -> None:
        self.config = config
        
    def _get_sampler(self) -> optuna.samplers.BaseSampler:
        """Factory method for Optuna Samplers based on config."""
        name = self.config.optimization.sampler
        seed = 42 # Could be exposed in config
        
        if name == "TPESampler":
            return optuna.samplers.TPESampler(seed=seed)
        elif name == "RandomSampler":
            return optuna.samplers.RandomSampler(seed=seed)
        elif name == "CmaEsSampler":
            return optuna.samplers.CmaEsSampler(seed=seed)
        elif name == "GridSampler":
            # GridSampler requires search space to be known at init, 
            # which is complex for this dynamic setup. defaulting to TPE.
            print("Warning: GridSampler not fully supported in dynamic mode. Using TPE.")
            return optuna.samplers.TPESampler(seed=seed)
        else:
            raise ValueError(f"Unsupported sampler: {name}")

    def _get_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        """Factory method for Optuna Pruners based on config."""
        name = self.config.optimization.pruner
        
        if not name or name == "None" or name == "NopPruner":
            return optuna.pruners.NopPruner()
        elif name == "MedianPruner":
            return optuna.pruners.MedianPruner()
        elif name == "HyperbandPruner":
            return optuna.pruners.HyperbandPruner()
        else:
            print(f"Warning: Pruner {name} not found, defaulting to NopPruner.")
            return optuna.pruners.NopPruner()

    def _get_direction(self) -> str:
        """Maps 'max'/'min' to 'maximize'/'minimize'."""
        mode = self.config.optimization.objective_mode
        if mode == "max":
            return "maximize"
        elif mode == "min":
            return "minimize"
        else:
            # Fallback based on metric name convention
            metric = self.config.optimization.objective_metric
            if "loss" in metric or "error" in metric:
                return "minimize"
            return "maximize"

    def objective(self, trial: optuna.Trial) -> float:
        # 1. Suggest Params (This logic ideally reads from a search space file, 
        # but here we demonstrate modifying the config objects)
        lr_min, lr_max = self.config.optimization.lr_range
        dropout_min, dropout_max = self.config.optimization.dropout_range
        lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
        dropout = trial.suggest_float("dropout", dropout_min, dropout_max)
        model_name, loss_name, activation_name, optimizer_name, tile_px, tile_mpp, feat_extractor = (
            self._suggest_search_space(trial)
        )
        
        # Apply to Config (Temporary for this trial)
        self.config.mil.lr = lr
        self.config.mil.dropout_p = dropout
        self.config.mil.best_epoch_based_on = self.config.optimization.objective_metric
    

        setattr(self.config, "_active_model_name", model_name)
        if loss_name is not None:
            setattr(self.config, "_active_loss_name", loss_name)
        if activation_name is not None:
            setattr(self.config, "_active_activation_name", activation_name)
        if optimizer_name is not None:
            setattr(self.config, "_active_optimizer_name", optimizer_name)
        if tile_px is not None:
            setattr(self.config, "_active_tile_px", tile_px)
        if tile_mpp is not None:
            setattr(self.config, "_active_tile_mpp", tile_mpp)
        if feat_extractor is not None:
            setattr(self.config, "_active_feature_extraction", feat_extractor)

        # 2. Instantiate Components
        # Use the registries and abstract factories
        trainer_backend = self.config.experiment.trainer_backend
        TrainerClass = TRAINERS.get(trainer_backend)

        ModelClass = MODELS.get(model_name)

        output_dim = self._resolve_output_dim()
        model = ModelClass(input_dim=1024, dropout=dropout, output_dim=output_dim)

        loss_name = self._resolve_loss_name()
        LossClass = LOSSES.get(loss_name)
        
        model = ModelClass(input_dim=1024, dropout=dropout, output_dim=2)
        loss_fn = LossClass()
        
        # 3. Data Loading
        train_entry = next(
            (ds for ds in self.config.datasets if ds.used_for == "training"),
            self.config.datasets[0],
        )
        val_entry = next(
            (ds for ds in self.config.datasets if ds.used_for == "validation"),
            self.config.datasets[1] if len(self.config.datasets) > 1 else None,
        )

        ds_train = BagDataset.from_config(train_entry, self.config)
        ds_val = BagDataset.from_config(val_entry, self.config) if val_entry else None

        # 4. Training
        # We can optionally pass an Optuna Pruning Callback here if the Trainer supports it
        from optuna.integration import PyTorchLightningPruningCallback
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor=self.config.optimization.objective_metric)
        
        trainer: TrainerBase
        if trainer_backend == "lightning":
            from optuna.integration import PyTorchLightningPruningCallback

            pruning_callback = PyTorchLightningPruningCallback(
                trial, monitor=self.config.optimization.objective_metric
            )
            trainer = TrainerClass(self.config, extra_callbacks=[pruning_callback])
        else:
            trainer = TrainerClass(self.config)

        
        try:
            result = trainer.fit(model, ds_train, ds_val, loss_fn)
            return result.best_score
        except Exception as e:
            # Handle pruning or NaN errors
            print(f"Trial failed: {e}")
            # Return worst possible score depending on direction
            return float('inf') if self._get_direction() == "minimize" else float('-inf')


    def _resolve_loss_name(self) -> str:
        if self.config.optimization.loss_name:
            return self.config.optimization.loss_name
        task = self.config.experiment.task
        if task == "regression":
            return "MSELoss"
        if task == "survival":
            return "CoxPHLoss"
        if task == "survival_discrete":
            return "DiscreteTimeNLLLoss"
        return "CrossEntropyLoss"

    def _suggest_search_space(
        self, trial: optuna.Trial
    ) -> tuple[
        str,
        str | None,
        str | None,
        str | None,
        int | None,
        float | None,
        str | None,
    ]:
        """
        Suggest trial parameters across the configured search space.

        Returns:
            model_name, loss_name, activation_name, optimizer_name,
            tile_px, tile_mpp, feature_extraction
        """
        search_space = self.config.search_space
        if not search_space.mil:
            raise ValueError("Search space is missing MIL models for optimization.")
        model_name = trial.suggest_categorical("model", search_space.mil)

        loss_name = None
        if search_space.loss:
            loss_name = trial.suggest_categorical("loss", search_space.loss)

        activation_name = None
        if search_space.activation_function:
            activation_name = trial.suggest_categorical(
                "activation", search_space.activation_function
            )

        optimizer_name = None
        if search_space.optimizer:
            optimizer_name = trial.suggest_categorical("optimizer", search_space.optimizer)

        tile_px = None
        if search_space.tile_px:
            tile_px = trial.suggest_categorical("tile_px", search_space.tile_px)

        tile_mpp = None
        if search_space.tile_mpp:
            tile_mpp = trial.suggest_categorical("tile_mpp", search_space.tile_mpp)

        feature_extraction = None
        if search_space.feature_extraction:
            feature_extraction = trial.suggest_categorical(
                "feature_extraction", search_space.feature_extraction
            )

        return (
            model_name,
            loss_name,
            activation_name,
            optimizer_name,
            tile_px,
            tile_mpp,
            feature_extraction,
        )

    def _resolve_output_dim(self) -> int:
        task = self.config.experiment.task
        if task in {"regression", "survival"}:
            return 1
        return max(self.config.mil.k, 1)
        
    def execute(self) -> None:
        print(f"Starting Optimization Study: {self.config.optimization.study_name}")
        
        sampler = self._get_sampler()
        pruner = self._get_pruner()
        direction = self._get_direction()
        
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            study_name=self.config.optimization.study_name,
            load_if_exists=self.config.optimization.load_study
        )
        
        study.optimize(self.objective, n_trials=self.config.optimization.trials)
        
        print(f"Best Params: {study.best_params}")
        print(f"Best Value ({self.config.optimization.objective_metric}): {study.best_value}")
        
        # Save results
        df = study.trials_dataframe()
        df.to_csv(f"{self.config.optimization.study_name}_results.csv")