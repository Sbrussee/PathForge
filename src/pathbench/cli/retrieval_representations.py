from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import dask
import numpy as np
import typer
from torch.utils.data import DataLoader

from ..benchmarking.tasks.slide_retrieval import (
    SlideRetrievalTask,
    _retrieval_batch_collate,
)
from ..config.config import Config
from ..core.experiments.base import Experiment
from ..core.experiments.combo_ids import build_feature_name, build_tiling_id
from ..core.experiments.combinations import ComboConfig, build_combinations
from ..core.features.utils import find_slides_with_missing_features
from ..core.io.slide_artifacts import tiles as tiles_io
from ..core.io.slide_artifacts.base import FileHandleH5
from ..policy.benchmarking import BenchmarkingPolicy
from ..slide_retrieval.representation_strategies.mean_rgb import resolve_sample_patch_mean_rgb
from ..slide_retrieval.representation_strategies.registry import (
    build_representation_strategy,
)
from ..slide_retrieval.representation_strategies.storage import (
    build_retrieval_representation_id,
)
from ..utils.constants import DATASET_COL, SLIDE_ID_COL
from .common import LOG_LEVEL_CHOICES, configure_logging

logger = logging.getLogger(__name__)
_VALID_RETRIEVAL_USES = {"reference", "query", "query_reference"}
_RGB_MEAN_REPRESENTATIONS = {"yottixel-rgb", "splice-rgb"}
_MISSING_MEAN_RGB_SLIDE_ERROR = (
    "Missing stored patch mean RGB descriptors and no source slide is available"
)


def _build_cli_sample_loader(
    *,
    representation_name: str,
    representation_strategy: Any,
    task: SlideRetrievalTask,
) -> Any:
    if representation_name not in _RGB_MEAN_REPRESENTATIONS:
        return representation_strategy.load_sample

    cfg = getattr(task.experiment, "cfg", None)

    def _load_sample_for_rgb(
        *,
        index: int,
        sample: Any,
        base_dataset: Any,
    ) -> dict[str, Any]:
        _ = index
        bag_id = str(base_dataset.tiling_id)
        mean_rgb = resolve_sample_patch_mean_rgb(
            sample=sample,
            bag_id=bag_id,
            config=cfg,
        )

        coord_parts: list[np.ndarray] = []
        for artifact_path in sample.artifact_paths:
            with FileHandleH5(artifact_path, mode="r") as slide_artifact:
                coords = tiles_io.read_coords(slide_artifact, bag_id=bag_id)
            coord_parts.append(np.asarray(coords[:, :2], dtype=np.int64))

        return {
            "mean_rgb": np.asarray(mean_rgb, dtype=np.float32),
            "coords": (
                np.concatenate(coord_parts, axis=0)
                if coord_parts
                else np.empty((0, 2), dtype=np.int64)
            ),
            "tiling_id": bag_id,
        }

    return _load_sample_for_rgb


def _is_missing_mean_rgb_slide_error(error_text: str) -> bool:
    return _MISSING_MEAN_RGB_SLIDE_ERROR in str(error_text)


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
    loader_workers = task._resolve_representation_loader_workers(
        representation_strategy=representation_strategy,
        default_workers=num_workers,
    )
    materialization_workers = task._resolve_representation_workers(
        representation_strategy=representation_strategy,
        default_workers=max(1, num_workers),
    )
    retrieval_batch_size = max(1, materialization_workers)
    representation_id = build_retrieval_representation_id(
        feature_extraction=feature_name,
        retrieval_representation=representation_name,
        params=representation_strategy.hyperparam_values(),
    )

    failed_creation_errors: dict[str, str] = {}
    total_cached_count = 0
    total_missing_count = 0
    total_created_count = 0
    total_skipped_missing_descriptors = 0
    sample_loader = _build_cli_sample_loader(
        representation_name=representation_name,
        representation_strategy=representation_strategy,
        task=task,
    )

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

            bag_dataset.bind_sample_loader(sample_loader)
            try:
                retrieval_loader = DataLoader(
                    missing_subset,
                    batch_size=retrieval_batch_size,
                    shuffle=False,
                    num_workers=loader_workers,
                    collate_fn=_retrieval_batch_collate,
                )
                created_retrieval_representations, creation_errors_by_sample = (
                    task.compute_retrieval_representations(
                        bag_dataset=bag_dataset,
                        retrieval_loader=retrieval_loader,
                        batch_thread_workers=materialization_workers,
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
            tolerated_errors = {
                sample_id: error_text
                for sample_id, error_text in creation_errors_by_sample.items()
                if _is_missing_mean_rgb_slide_error(error_text)
            }
            if tolerated_errors:
                total_skipped_missing_descriptors += len(tolerated_errors)
                logger.warning(
                    "[SlideRetrieval] Skipping %d sample(s) with missing mean_rgb "
                    "descriptors and no available source slide.",
                    len(tolerated_errors),
                )

            failed_creation_errors.update(
                {
                    sample_id: error_text
                    for sample_id, error_text in creation_errors_by_sample.items()
                    if sample_id not in tolerated_errors
                }
            )

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
        "num_skipped_missing_descriptors": total_skipped_missing_descriptors,
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
