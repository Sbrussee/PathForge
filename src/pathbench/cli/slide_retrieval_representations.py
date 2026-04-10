from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import dask
from torch.utils.data import DataLoader

from ..benchmarking.tasks.slide_retrieval import (
    SlideRetrievalTask,
    _retrieval_batch_collate,
)
from ..config.config import Config
from ..core.experiments.base import Experiment
from ..core.experiments.combo_ids import build_feature_name, build_tiling_id
from ..core.experiments.combinations import ComboConfig, build_combinations
from ..policy.benchmarking import BenchmarkingPolicy
from ..slide_retrieval.representation_strategies.registry import (
    build_representation_strategy,
)
from ..slide_retrieval.representation_strategies.storage import (
    build_retrieval_representation_id,
)

logger = logging.getLogger(__name__)
_VALID_RETRIEVAL_USES = {"reference", "query", "query_reference"}


def _materialize_representations_for_combo(
    *,
    task: SlideRetrievalTask,
    combo_cfg: ComboConfig,
    datasets_by_use: dict[str, list[Any]],
) -> dict[str, Any]:
    """Run only stage 1+2 of SlideRetrievalTask for one combo."""
    tiling_id = build_tiling_id(combo_cfg)
    aggregation_level = str(task.cfg.experiment.aggregation_level)
    task._validate_dataset_context(
        datasets_by_use=datasets_by_use,
        tiling_id=tiling_id,
        aggregation_level=aggregation_level,
    )
    feature_name = build_feature_name(combo_cfg)
    exclusion_level = task._resolve_exclusion_level()
    representation_name = str(combo_cfg.get("retrieval_representation"))
    search_strategy_name = str(combo_cfg.get("search_strategy"))
    num_workers = int(getattr(task.cfg.experiment, "num_workers", 0) or 0)
    retrieval_batch_size = max(1, num_workers)

    combination_is_valid, reason = task._validate_combination_compatibility(
        datasets_by_use=datasets_by_use,
        representation_name=representation_name,
        search_strategy_name=search_strategy_name,
        aggregation_level=aggregation_level,
        exclusion_level=exclusion_level,
    )
    if not combination_is_valid:
        logger.warning("[SlideRetrieval] Skipping combo: %s", reason)
        return {"status": "skipped_incompatible_combo", "reason": reason}

    representation_strategy = build_representation_strategy(
        representation_name,
        params=combo_cfg.get_hyperparams("retrieval_representation"),
        bag_id=tiling_id,
        config=getattr(task.experiment, "cfg", None),
    )
    representation_id = build_retrieval_representation_id(
        feature_extraction=feature_name,
        retrieval_representation=representation_name,
        params=representation_strategy.hyperparam_values(),
    )

    failed_creation_errors: dict[str, str] = {}
    total_cached_count = 0
    total_missing_count = 0
    total_created_count = 0

    for use, bag_datasets in datasets_by_use.items():
        if use not in _VALID_RETRIEVAL_USES:
            raise ValueError(
                f"Unsupported retrieval dataset use '{use}'. "
                f"Expected one of: {sorted(_VALID_RETRIEVAL_USES)}"
            )

        for bag_dataset in bag_datasets:
            existing_representations, missing_subset = task._collect_existing_representations(
                bag_dataset=bag_dataset,
                representation_id=representation_id,
                aggregation_level=aggregation_level,
                exclusion_level=exclusion_level,
            )
            total_cached_count += len(existing_representations)
            missing_count = 0 if missing_subset is None else len(missing_subset)
            total_missing_count += missing_count

            if missing_subset is None:
                continue

            bag_dataset.bind_sample_loader(representation_strategy.load_sample)
            try:
                retrieval_loader = DataLoader(
                    missing_subset,
                    batch_size=retrieval_batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=_retrieval_batch_collate,
                )
                created_retrieval_representations, creation_errors_by_sample = (
                    task.compute_retrieval_representations(
                        bag_dataset=bag_dataset,
                        retrieval_loader=retrieval_loader,
                        batch_thread_workers=retrieval_batch_size,
                        combo_cfg=combo_cfg,
                        representation_strategy=representation_strategy,
                        representation_id=representation_id,
                        aggregation_level=aggregation_level,
                        exclusion_level=exclusion_level,
                    )
                )
            finally:
                bag_dataset.clear_sample_loader()

            total_created_count += len(created_retrieval_representations)
            failed_creation_errors.update(creation_errors_by_sample)

    if failed_creation_errors:
        failed_items = ", ".join(sorted(failed_creation_errors))
        error_details = "\n".join(
            f"- {sample_id}: {error_text}"
            for sample_id, error_text in sorted(failed_creation_errors.items())
        )
        raise RuntimeError(
            "Slide retrieval representation creation failed for one or more "
            f"samples: {failed_items}\nRoot errors:\n{error_details}"
        )

    return {
        "status": "representations_ready",
        "representation_id": representation_id,
        "num_cached": total_cached_count,
        "num_planned_new": total_missing_count,
        "num_created": total_created_count,
    }


def _run_representation_precompute(policy: BenchmarkingPolicy) -> dict[str, Any]:
    task = policy.task
    if not isinstance(task, SlideRetrievalTask):
        raise TypeError(
            "slide_retrieval_representations CLI requires the slide_retrieval task."
        )

    combinations = build_combinations(
        cfg=policy.experiment.cfg,
        keys=task.get_grid_keys(),
    )
    if not combinations:
        logger.warning("[Benchmark] No benchmark combinations found.")
        return {"status": "no_combos", "num_runs": 0}

    combinations_by_bag_id = policy._group_combos_by_bag_source(combinations)
    annotations_df = policy.experiment.load_annotations()
    num_runs = 0
    for bag_id, combinations_for_bag_id in combinations_by_bag_id.items():
        bag_source_combo = combinations_for_bag_id[0]
        logger.info("[Benchmark] Representation-only bag group | bag_id=%s", bag_id)
        policy.ensure_bag_features_exist(
            combo_cfg=bag_source_combo,
            annotations_df=annotations_df,
        )
        bag_datasets = policy.build_bag_datasets_for_combo(
            combo_cfg=bag_source_combo,
            annotations_df=annotations_df,
        )
        datasets_by_use = policy.group_bag_datasets_by_use(bag_datasets)
        policy._validate_dataset_uses(datasets_by_use=datasets_by_use)
        for full_combo_cfg in combinations_for_bag_id:
            _materialize_representations_for_combo(
                task=task,
                combo_cfg=full_combo_cfg,
                datasets_by_use=datasets_by_use,
            )
            num_runs += 1

    return {"status": "representations_done", "num_runs": num_runs}


def main(argv: list[str] | None = None) -> int:
    """Run only the slide-retrieval representation materialization stage."""
    parser = argparse.ArgumentParser(
        description="Slide retrieval representation precompute workflow",
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info("Starting slide-retrieval representation precompute CLI")
    logger.info("Using config: %s", args.config)

    dask.config.set({"dataframe.query-planning": True})

    cfg = Config.from_yaml(args.config)
    if cfg.experiment.mode != "benchmark":
        raise ValueError(
            "Representation precompute CLI requires experiment.mode='benchmark'. "
            f"Got {cfg.experiment.mode!r}."
        )
    if cfg.experiment.task != "slide_retrieval":
        raise ValueError(
            "Representation precompute CLI requires experiment.task='slide_retrieval'. "
            f"Got {cfg.experiment.task!r}."
        )

    experiment = Experiment(cfg)
    policy = BenchmarkingPolicy(experiment)

    output = _run_representation_precompute(policy)
    logger.info("Representation precompute finished with status: %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
