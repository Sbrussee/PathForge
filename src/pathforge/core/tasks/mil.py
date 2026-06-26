from __future__ import annotations

import copy
from typing import Any

from pathforge.core.tasks.registry import register_task
from pathforge.core.tasks.base import TaskBase
from pathforge.core.datasets.bag_dataset import BagDataset
from pathforge.core.experiments.combinations import ComboConfig
from pathforge.policy.utils import apply_search_params
from pathforge.policy.utils import build_mil_model_for_config
from pathforge.policy.utils import infer_model_dimensions
from pathforge.utils.registries import LOSSES
from pathforge.utils.registries import TRAINERS

_MIL_DATASET_USES = frozenset({"training", "validation", "testing", "all"})


class _BaseMilTask(TaskBase):
    grid_keys = ["feature_extraction", "tile_px", "tile_mpp", "mil", "loss"]
    allowed_dataset_uses = _MIL_DATASET_USES

    def execute(
        self,
        combo_cfg: ComboConfig,
        datasets_by_use: dict[str, list[BagDataset]],
    ) -> dict[str, Any]:
        run_cfg = copy.deepcopy(self.cfg)
        combo_params = {
            key: value
            for key, value in combo_cfg.to_dict().items()
            if not key.endswith("_params")
        }
        apply_search_params(run_cfg, combo_params)

        train_dataset, val_dataset = self._resolve_train_val_datasets(datasets_by_use)
        input_dim, output_dim = infer_model_dimensions(train_dataset)
        model_name = getattr(run_cfg, "_active_model_name", run_cfg.benchmark_parameters.mil[0])
        loss_name = getattr(run_cfg, "_active_loss_name", run_cfg.benchmark_parameters.loss[0])
        model = build_mil_model_for_config(
            run_cfg,
            model_name=model_name,
            input_dim=input_dim,
            output_dim=output_dim,
            extra_kwargs={"dropout": run_cfg.mil.dropout_p},
        )
        trainer = TRAINERS.get("lightning")(run_cfg)
        loss_fn = LOSSES.get(loss_name)()
        best_path, best_score = trainer.fit(model, train_dataset, val_dataset, loss_fn)
        return {
            "status": "success",
            "checkpoint_path": best_path,
            "objective_value": best_score,
        }

    def _resolve_train_val_datasets(
        self,
        datasets_by_use: dict[str, list[BagDataset]],
    ) -> tuple[BagDataset, BagDataset]:
        training = datasets_by_use.get("training") or datasets_by_use.get("all") or []
        validation = (
            datasets_by_use.get("validation")
            or datasets_by_use.get("testing")
            or datasets_by_use.get("all")
            or []
        )
        if not training:
            raise ValueError("MIL benchmarking requires at least one training or all-use dataset.")
        if not validation:
            validation = training
        return training[0], validation[0]


@register_task("classification")
class ClassificationMilTask(_BaseMilTask):
    """Standard MIL classification benchmarking task."""


@register_task("regression")
class RegressionMilTask(_BaseMilTask):
    """Standard MIL regression benchmarking task."""


@register_task("survival")
class SurvivalMilTask(_BaseMilTask):
    """Standard MIL continuous-survival benchmarking task."""


@register_task("survival_discrete")
class SurvivalDiscreteMilTask(_BaseMilTask):
    """Standard MIL discrete-survival benchmarking task."""
