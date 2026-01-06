from __future__ import annotations
from typing import Any, Dict, List
import itertools
import pandas as pd
import numpy as np
import copy
import logging
from pathlib import Path
from pathbench.policy.base import PolicyBase, ExperimentLike
from pathbench.utils.registries import MODELS, LOSSES, TRAINERS
from pathbench.core.datasets.bag_dataset import BagDataset
from pathbench.training.base import TrainerBase
from pathbench.training.metrics import evaluate_predictions

class BenchmarkingPolicy(PolicyBase):
    """
    Executes grid search over configurations.
    Strictly uses TrainerBase abstraction.
    """

    def __init__(self, experiment: ExperimentLike):
        super().__init__(experiment)
        self.config = self.cfg
        self.results: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("pathbench.benchmark")

    def _generate_configs(self) -> List[Config]:
        bp = self.config.search_space
        grid = {
            "mil_model": bp.mil,
            "loss": bp.loss,
            "seed": [1, 2, 3] 
        }
        keys, values = zip(*grid.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        config_objects = []
        for exp in experiments:
            new_cfg = copy.deepcopy(self.config)
            setattr(new_cfg, "_active_model_name", exp["mil_model"])
            setattr(new_cfg, "_active_loss_name", exp["loss"])
            config_objects.append(new_cfg)
        return config_objects

    def execute(self) -> dict[str, Any]:
        configs_to_run = self._generate_configs()
        
        trainer_backend = self.config.experiment.trainer_backend
        TrainerClass = TRAINERS.get(trainer_backend)
        
        if not TrainerClass:
             raise ValueError(f"Trainer backend '{trainer_backend}' not found.")

        for i, run_cfg in enumerate(configs_to_run):
            model_name = getattr(run_cfg, "_active_model_name")
            loss_name = getattr(run_cfg, "_active_loss_name")
            
            print(f"Benchmarking {model_name} with {loss_name}...")
            
            try:
                # 1. Components
                ModelClass = MODELS.get(model_name)
                LossClass = LOSSES.get(loss_name)
                output_dim = _resolve_output_dim()
                
                # 2. Data
                train_entry = next(
                    (ds for ds in run_cfg.datasets if ds.used_for == "training"),
                    run_cfg.datasets[0],
                )
                val_entry = next(
                    (ds for ds in run_cfg.datasets if ds.used_for == "validation"),
                    run_cfg.datasets[1] if len(run_cfg.datasets) > 1 else None,
                )
                ds_train = BagDataset.from_config(train_entry, run_cfg)
                ds_val = BagDataset.from_config(val_entry, run_cfg) if val_entry else None

                input_dim = self._resolve_input_dim(ds_train)
                model = ModelClass(input_dim=input_dim, output_dim=output_dim)
                loss_fn = LossClass()

                # 3. Abstract Trainer Instantiation
                trainer: TrainerBase = TrainerClass(run_cfg)
                
                # 4. Fit
                trainer.fit(model, ds_train, ds_val, loss_fn)
                
                # 5. Log
                metrics_out: Dict[str, float] = {}
                if ds_val is not None:
                    preds = trainer.predict(model, ds_val)
                    metrics_out = evaluate_predictions(
                        preds,
                        ds_val.labels,
                        run_cfg.experiment.task or "classification",
                        run_cfg.evaluation.metrics,
                        run_cfg.evaluation.average,
                        run_cfg.evaluation.positive_label,
                    )
    

                self.results.append({
                    "model": model_name,
                    "loss": loss_name,
                    "status": "success",
                    "metrics": metrics_out,
                })
                

            except Exception as e:
                self.logger.error(f"Run failed: {e}")
                self.results.append({"model": model_name, "error": str(e)})

        self._save_report()
        return {"status": "benchmark_complete", "results_file": "benchmark_results.csv"}

    def _resolve_output_dim(self) -> int:
        task = self.config.experiment.task
        if task in {"regression", "survival"}:
            return 1
        return max(self.config.mil.k, 1)

    def _resolve_input_dim(self, dataset: BagDataset) -> int:
        """
        Infer input feature dimension from a dataset's stored bags.

        Args:
            dataset: BagDataset instance to inspect.

        Returns:
            Feature dimension inferred from stored bag tensors.
        """
        return dataset.infer_feature_dim()

    def _save_report(self):
        df = pd.DataFrame(self.results)
        df.to_csv("benchmark_results.csv", index=False)