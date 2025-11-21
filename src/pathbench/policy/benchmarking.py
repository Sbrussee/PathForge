from __future__ import annotations
from typing import Any, Dict, List
import itertools
import pandas as pd
import copy
import logging
from pathlib import Path

from pathbench.policy.base import PolicyBase
from pathbench.config.config import Config
from pathbench.utils.registries import MODELS, LOSSES, TRAINERS
from pathbench.core.datasets.bag_dataset import BagDataset
from pathbench.training.base import TrainerBase

class BenchmarkingPolicy(PolicyBase):
    """
    Executes grid search over configurations.
    Strictly uses TrainerBase abstraction.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.results: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("pathbench.benchmark")

    def _generate_configs(self) -> List[Config]:
        # ... (Same as previous implementation) ...
        bp = self.config.benchmark_parameters
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
                # 1. Components
                ModelClass = MODELS.get(model_name)
                LossClass = LOSSES.get(loss_name)
                model = ModelClass(input_dim=1024, output_dim=2)
                loss_fn = LossClass()
                
                # 2. Data
                ds_train = BagDataset("train", run_cfg.datasets[0].tile_path, run_cfg.experiment.annotation_file, "label")
                ds_val = BagDataset("val", run_cfg.datasets[1].tile_path, run_cfg.experiment.annotation_file, "label")

                # 3. Abstract Trainer Instantiation
                trainer: TrainerBase = TrainerClass(run_cfg)
                
                # 4. Fit
                trainer.fit(model, ds_train, ds_val, loss_fn)
                
                # 5. Log
                self.results.append({
                    "model": model_name,
                    "loss": loss_name,
                    "status": "success"
                })
                
            except Exception as e:
                self.logger.error(f"Run failed: {e}")
                self.results.append({"model": model_name, "error": str(e)})

        self._save_report()

    def _save_report(self):
        df = pd.DataFrame(self.results)
        df.to_csv("benchmark_results.csv", index=False)