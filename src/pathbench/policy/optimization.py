from __future__ import annotations
from typing import Optional
import optuna
import pandas as pd
from pathbench.policy.base import PolicyBase
from pathbench.config.config import Config
from pathbench.utils.registries import LOSSES, TRAINERS
from pathbench.training.base import TrainerBase
from pathbench.policy.utils import (
    apply_search_params,
    build_bag_dataset_for_task,
    build_mil_model_for_config,
    experiment_output_root,
    infer_model_dimensions,
    optimization_search_space,
    resolve_dataset_feature_dir,
    save_optuna_visualizations,
    suggest_parameter,
    write_experiment_summary_csv,
)


class OptimizationPolicy(PolicyBase):
    """
    Optimization Policy that dynamically configures Optuna based on Config.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = self.cfg

    def _get_sampler(self) -> optuna.samplers.BaseSampler:
        """Factory method for Optuna Samplers based on config."""
        name = self.config.optimization.sampler
        seed = 42  # Could be exposed in config

        if name == "TPESampler":
            return optuna.samplers.TPESampler(seed=seed)
        elif name == "RandomSampler":
            return optuna.samplers.RandomSampler(seed=seed)
        elif name == "CmaEsSampler":
            return optuna.samplers.CmaEsSampler(seed=seed)
        elif name == "GridSampler":
            # GridSampler requires search space to be known at init,
            # which is complex for this dynamic setup. defaulting to TPE.
            print(
                "Warning: GridSampler not fully supported in dynamic mode. Using TPE."
            )
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
        suggested_params = {
            name: suggest_parameter(trial, name=name, spec=spec)
            for name, spec in optimization_search_space(self.config).items()
        }
        apply_search_params(self.config, suggested_params)
        self.config.mil.best_epoch_based_on = self.config.optimization.objective_metric
        model_name = getattr(
            self.config,
            "_active_model_name",
            self.config.benchmark_parameters.mil[0],
        )
        loss_name = getattr(
            self.config,
            "_active_loss_name",
            self.config.benchmark_parameters.loss[0],
        )

        # 2. Instantiate Components
        # Use the registries and abstract factories
        TrainerClass = TRAINERS.get("lightning")
        LossClass = LOSSES.get(loss_name)

        train_entry = self.config.datasets[0]
        val_entry = self.config.datasets[min(1, len(self.config.datasets) - 1)]
        ds_train = build_bag_dataset_for_task(
            self.config,
            feature_dir=resolve_dataset_feature_dir(train_entry),
            name="train",
        )
        ds_val = build_bag_dataset_for_task(
            self.config,
            feature_dir=resolve_dataset_feature_dir(val_entry),
            name="val",
        )
        input_dim, output_dim = infer_model_dimensions(ds_train)
        model = build_mil_model_for_config(
            self.config,
            model_name=model_name,
            input_dim=input_dim,
            output_dim=output_dim,
            extra_kwargs={"dropout": self.config.mil.dropout_p},
        )
        loss_fn = LossClass()

        # 4. Training
        # We can optionally pass an Optuna Pruning Callback here if the Trainer supports it
        from optuna.integration import PyTorchLightningPruningCallback

        pruning_callback = PyTorchLightningPruningCallback(
            trial, monitor=self.config.optimization.objective_metric
        )

        trainer: TrainerBase = TrainerClass(
            self.config, extra_callbacks=[pruning_callback]
        )

        try:
            best_path, best_score = trainer.fit(model, ds_train, ds_val, loss_fn)
            return best_score
        except Exception as e:
            # Handle pruning or NaN errors
            print(f"Trial failed: {e}")
            # Return worst possible score depending on direction
            return (
                float("inf") if self._get_direction() == "minimize" else float("-inf")
            )

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
            load_if_exists=self.config.optimization.load_study,
        )

        study.optimize(self.objective, n_trials=self.config.optimization.trials)

        print(f"Best Params: {study.best_params}")
        print(
            f"Best Value ({self.config.optimization.objective_metric}): {study.best_value}"
        )

        self._save_study_outputs(study)

    def _save_study_outputs(self, study: optuna.Study) -> None:
        output_root = experiment_output_root(self.config)
        raw_results_path = output_root / f"{self.config.optimization.study_name}_results.csv"
        summary_path = output_root / "optimization_results.csv"
        raw_df = study.trials_dataframe()
        raw_df.to_csv(raw_results_path, index=False)

        rows = self._build_summary_rows(raw_df)
        write_experiment_summary_csv(
            rows,
            output_path=summary_path,
            objective_metric=self.config.optimization.objective_metric,
            minimize=self._get_direction() == "minimize",
        )
        save_optuna_visualizations(
            study,
            output_dir=output_root / "optimization_visualizations",
        )

    def _build_summary_rows(self, trials_df: pd.DataFrame) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        objective_metric = self.config.optimization.objective_metric
        for _, trial in trials_df.iterrows():
            row = {
                "run_index": int(trial["number"]),
                "project_name": self.config.experiment.project_name,
                "mode": self.config.experiment.mode,
                "task": self.config.experiment.task,
                "status": str(trial.get("state", "")),
                "objective_metric": objective_metric,
                "objective_value": trial.get("value"),
                "trial_number": int(trial["number"]),
            }
            for column, value in trial.items():
                if column in row:
                    continue
                row[str(column)] = value
            rows.append(row)
        return rows
