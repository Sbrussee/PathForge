from __future__ import annotations
from typing import Any, Dict, List
import itertools
import pandas as pd
import copy
import logging

from pathbench.policy.base import PolicyBase
from pathbench.config.config import Config
from pathbench.utils.registries import LOSSES, TRAINERS
from pathbench.training.base import TrainerBase
from pathbench.policy.utils import (
    apply_search_params,
    benchmark_search_space,
    build_bag_dataset_for_task,
    build_mil_model_for_config,
    infer_model_dimensions,
    resolve_dataset_feature_dir,
)


class BenchmarkingPolicy(PolicyBase):
    """
    Executes grid search over configurations.
    Strictly uses TrainerBase abstraction.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config = self.cfg
        self.results: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("pathbench.benchmark")

    def _generate_configs(self) -> List[Config]:
        grid = benchmark_search_space(self.config)
        keys, values = zip(*grid.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        config_objects = []
        for exp in experiments:
            new_cfg = copy.deepcopy(self.config)
            config_objects.append(apply_search_params(new_cfg, exp))
        return config_objects

    def execute(self) -> None:
        configs_to_run = self._generate_configs()

        # Resolve Trainer Class once (assuming same trainer for all)
        # We default to 'lightning' if not specified, or add a field to ExperimentConfig
        trainer_backend = "lightning"
        TrainerClass = TRAINERS.get(trainer_backend)

        if not TrainerClass:
            raise ValueError(f"Trainer backend '{trainer_backend}' not found.")

        for i, run_cfg in enumerate(configs_to_run):
            model_name = getattr(run_cfg, "_active_model_name")
            loss_name = getattr(run_cfg, "_active_loss_name")

            print(f"Benchmarking {model_name} with {loss_name}...")

            try:
                # 1. Data + dimensions
                train_entry = run_cfg.datasets[0]
                val_entry = run_cfg.datasets[min(1, len(run_cfg.datasets) - 1)]
                ds_train = build_bag_dataset_for_task(
                    run_cfg,
                    feature_dir=resolve_dataset_feature_dir(train_entry),
                    name="train",
                )
                ds_val = build_bag_dataset_for_task(
                    run_cfg,
                    feature_dir=resolve_dataset_feature_dir(val_entry),
                    name="val",
                )
                input_dim, output_dim = infer_model_dimensions(ds_train)

                # 2. Components
                LossClass = LOSSES.get(loss_name)
                model = build_mil_model_for_config(
                    run_cfg,
                    model_name=model_name,
                    input_dim=input_dim,
                    output_dim=output_dim,
                )
                loss_fn = LossClass()

                # 3. Abstract Trainer Instantiation
                trainer: TrainerBase = TrainerClass(run_cfg)

                # 4. Fit
                trainer.fit(model, ds_train, ds_val, loss_fn)

                # 5. Log
                self.results.append(
                    {"model": model_name, "loss": loss_name, "status": "success"}
                )

            except Exception as e:
                self.logger.error(f"Run failed: {e}")
                self.results.append({"model": model_name, "error": str(e)})

        self._save_report()

    def _save_report(self):
        df = pd.DataFrame(self.results)
        df.to_csv("benchmark_results.csv", index=False)
