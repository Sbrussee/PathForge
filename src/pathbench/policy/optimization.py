from __future__ import annotations
from typing import Optional
import optuna
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
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        model_name = trial.suggest_categorical("model", ["AttentionMIL", "TransMIL"])
        
        # Apply to Config (Temporary for this trial)
        if self.config.classification is None:
            raise ValueError("classification config is required for optimization.")
        self.config.classification.lr = lr
        self.config.classification.dropout_p = dropout
        self.config.classification.best_epoch_based_on = self.config.optimization.objective_metric
        
        # 2. Instantiate Components
        # Use the registries and abstract factories
        TrainerClass = TRAINERS.get("lightning")
        ModelClass = MODELS.get(model_name)
        LossClass = LOSSES.get("CrossEntropyLoss") # parameterized if needed
        
        model = ModelClass(input_dim=1024, dropout=dropout, output_dim=2)
        loss_fn = LossClass()
        
        # 3. Data Loading (Mocking paths from config)
        # In production, ensure these paths are valid or passed via config
        ds_train = BagDataset("train", self.config.datasets[0].tile_path, self.config.experiment.annotation_file, "label")
        ds_val = BagDataset("val", self.config.datasets[1].tile_path, self.config.experiment.annotation_file, "label")
        
        # 4. Training
        # We can optionally pass an Optuna Pruning Callback here if the Trainer supports it
        from optuna.integration import PyTorchLightningPruningCallback
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor=self.config.optimization.objective_metric)
        
        trainer: TrainerBase = TrainerClass(self.config, extra_callbacks=[pruning_callback])
        
        try:
            best_path, best_score = trainer.fit(model, ds_train, ds_val, loss_fn)
            return best_score
        except Exception as e:
            # Handle pruning or NaN errors
            print(f"Trial failed: {e}")
            # Return worst possible score depending on direction
            return float('inf') if self._get_direction() == "minimize" else float('-inf')

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
