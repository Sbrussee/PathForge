from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import dask
import typer

from ..config.config import Config
from ..core.experiments.base import Experiment
from ..core.experiments.combinations import ComboConfig, build_combinations
from ..core.experiments.combo_ids import build_feature_name, build_tiling_id
from ..core.features.utils import find_slides_with_missing_features
from ..core.tasks.slide_retrieval import SlideRetrievalTask
from ..policy.benchmarking import BenchmarkingPolicy
from ..slide_retrieval.representation_strategies.registry import (
    build_representation_strategy,
)
from ..slide_retrieval.representation_strategies.storage import (
    build_retrieval_representation_id,
)
from ..utils.constants import DATASET_COL, SLIDE_ID_COL
from .common import LOG_LEVEL_CHOICES, configure_logging

logger = logging.getLogger(__name__)
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
    prepare_for_combo = getattr(representation_strategy, "prepare_for_combo", None)
    if callable(prepare_for_combo):
        prepare_for_combo(
            combo_cfg=combo_cfg,
            feature_name=feature_name,
            tiling_id=tiling_id,
        )
    representation_cache_params = task._representation_cache_params(
        representation_strategy
    )
    representation_id = build_retrieval_representation_id(
        feature_extraction=feature_name,
        retrieval_representation=representation_name,
        params=representation_cache_params,
    )

    representations_by_use = task._materialize_and_aggregate_representations(
        datasets_by_use=datasets_by_use,
        representation_strategy=representation_strategy,
        representation_id=representation_id,
        combo_cfg=combo_cfg,
        aggregation_level=aggregation_level,
        exclusion_level=exclusion_level,
        representation_cache_params=representation_cache_params,
    )
    total_created_count = sum(
        len(representations) for representations in representations_by_use.values()
    )

    return {
        "status": "representations_ready",
        "representation_id": representation_id,
        "num_cached": 0,
        "num_planned_new": total_created_count,
        "num_created": total_created_count,
        "num_skipped_missing_descriptors": 0,
    }


def _filter_annotations_with_existing_features(
    *,
    policy: BenchmarkingPolicy,
    combo_cfg: ComboConfig,
    annotations_df: Any,
) -> Any:
    """Return annotations filtered to rows whose required features already exist."""
    filtered_annotations = annotations_df.copy()
    allowed_uses = getattr(policy.task, "allowed_dataset_uses", None)

    for ds_cfg in policy.cfg.datasets:
        if ds_cfg.used_for == "ignore":
            continue
        if allowed_uses is not None and ds_cfg.used_for not in allowed_uses:
            continue

        missing_slide_ids = find_slides_with_missing_features(
            ds_cfg=ds_cfg,
            annotations_df=annotations_df,
            combo_cfg=combo_cfg,
        )
        if not missing_slide_ids:
            continue

        missing_slide_set = {str(slide_id) for slide_id in missing_slide_ids}
        drop_mask = (
            (filtered_annotations[DATASET_COL] == ds_cfg.name)
            & (filtered_annotations[SLIDE_ID_COL].astype(str).isin(missing_slide_set))
        )
        dropped_rows = int(drop_mask.sum())
        if dropped_rows > 0:
            logger.warning(
                "[Benchmark] Dataset '%s': skipping %d row(s) because features are missing.",
                ds_cfg.name,
                dropped_rows,
            )
            filtered_annotations = filtered_annotations.loc[~drop_mask].copy()

    return filtered_annotations


def _run_representation_precompute(
    policy: BenchmarkingPolicy,
    *,
    skip_missing_features: bool = False,
) -> dict[str, Any]:
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
    if not skip_missing_features:
        logger.info(
            "[Benchmark] Representation precompute is artifact-only; missing "
            "features will be skipped instead of triggering slide-based "
            "feature extraction."
        )
    for bag_id, combinations_for_bag_id in combinations_by_bag_id.items():
        bag_source_combo = combinations_for_bag_id[0]
        logger.info("[Benchmark] Representation-only bag group | bag_id=%s", bag_id)

        bag_annotations_df = _filter_annotations_with_existing_features(
            policy=policy,
            combo_cfg=bag_source_combo,
            annotations_df=annotations_df,
        )

        bag_datasets = policy.build_bag_datasets_for_combo(
            combo_cfg=bag_source_combo,
            annotations_df=bag_annotations_df,
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


def run_slide_retrieval_representations(
    *,
    config: Path,
    log_level: str = "INFO",
    skip_missing_features: bool = False,
) -> int:
    """Precompute slide-retrieval representations for one YAML config."""
    config_path = Path(config)
    configure_logging(log_level)
    logger.info("Starting slide-retrieval representation precompute CLI")
    logger.info("Using config: %s", config_path)

    dask.config.set({"dataframe.query-planning": True})

    cfg = Config.from_yaml(config_path)
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

    output = _run_representation_precompute(
        policy,
        skip_missing_features=bool(skip_missing_features),
    )
    logger.info("Representation precompute finished with status: %s", output)
    return 0


def run_command(
    config: Path = typer.Option(..., "--config", help="Path to YAML config"),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level.",
        show_default=True,
    ),
    skip_missing_features: bool = typer.Option(
        False,
        "--skip-missing-features",
        help=(
            "Deprecated compatibility flag. Representation precompute is now "
            "always artifact-only and skips slides with missing features."
        ),
    ),
) -> None:
    """Typer command that precomputes slide-retrieval representations from the provided options."""
    raise SystemExit(
        run_slide_retrieval_representations(
            config=config,
            log_level=log_level,
            skip_missing_features=skip_missing_features,
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Run only the slide-retrieval representation materialization stage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Slide retrieval representation precompute workflow",
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVEL_CHOICES,
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--skip-missing-features",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Representation precompute is now "
            "always artifact-only and skips slides with missing features."
        ),
    )
    args = parser.parse_args(argv)
    return run_slide_retrieval_representations(
        config=args.config,
        log_level=args.log_level,
        skip_missing_features=args.skip_missing_features,
    )


if __name__ == "__main__":
    raise SystemExit(main())
