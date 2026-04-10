from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

from pathbench.config.config import Config, DatasetEntry
from pathbench.core.experiments.combo_ids import build_tiling_id
from pathbench.core.experiments.combinations import ComboConfig
from pathbench.slide_retrieval.mean_rgb import resolve_sample_patch_mean_rgb


logger = logging.getLogger(__name__)


def _find_dataset_config(cfg: Config, dataset_name: str) -> DatasetEntry:
    for dataset_cfg in cfg.datasets:
        if str(dataset_cfg.name) == dataset_name:
            return dataset_cfg
    available_names = [str(dataset_cfg.name) for dataset_cfg in cfg.datasets]
    raise ValueError(
        f"Dataset '{dataset_name}' not found in config.datasets. Available: {available_names}"
    )


def _resolve_bag_ids(cfg: Config, explicit_bag_ids: list[str] | None) -> list[str]:
    if explicit_bag_ids:
        seen: set[str] = set()
        ordered: list[str] = []
        for bag_id in explicit_bag_ids:
            normalized = str(bag_id).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        if ordered:
            return ordered

    tile_px_values = cfg.benchmark_parameters.get_values("tile_px")
    tile_mpp_values = cfg.benchmark_parameters.get_values("tile_mpp")
    inferred: list[str] = []
    seen_inferred: set[str] = set()
    for tile_px in tile_px_values:
        for tile_mpp in tile_mpp_values:
            bag_id = build_tiling_id(
                ComboConfig(tile_px=int(tile_px), tile_mpp=float(tile_mpp))
            )
            if bag_id in seen_inferred:
                continue
            seen_inferred.add(bag_id)
            inferred.append(bag_id)
    return inferred


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute and persist slide retrieval mean_rgb descriptors for one slide "
            "and one or more bag IDs."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Dataset name (must exist in config.datasets).",
    )
    parser.add_argument(
        "--slide-id",
        required=True,
        type=str,
        help="Slide ID (without extension), used for source lookup and artifact naming.",
    )
    parser.add_argument(
        "--bag-id",
        action="append",
        default=None,
        help=(
            "Bag identifier (repeatable). If omitted, bag IDs are inferred from "
            "all tile_px/tile_mpp combinations in config.benchmark_parameters."
        ),
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit slide artifact path. Defaults to "
            "datasets[].artifacts_dir/{slide_id}.h5."
        ),
    )
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

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = Config.from_yaml(config_path)
    dataset_cfg = _find_dataset_config(cfg, dataset_name=str(args.dataset))
    slide_id = str(args.slide_id).strip()
    if not slide_id:
        raise ValueError("--slide-id must be a non-empty string.")

    artifact_path = (
        Path(args.artifact_path).expanduser().resolve()
        if args.artifact_path is not None
        else Path(dataset_cfg.artifacts_dir).expanduser().resolve() / f"{slide_id}.h5"
    )
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Slide artifact file not found for slide_id='{slide_id}': {artifact_path}"
        )

    bag_ids = _resolve_bag_ids(cfg, explicit_bag_ids=args.bag_id)
    if not bag_ids:
        raise ValueError(
            "No bag IDs provided and none could be inferred from config.benchmark_parameters "
            "(tile_px/tile_mpp)."
        )

    logger.info("Starting mean_rgb CLI")
    logger.info("Using config: %s", config_path)
    logger.info("Using dataset: %s", args.dataset)
    logger.info("Using slide_id: %s", slide_id)
    logger.info("Using artifact: %s", artifact_path)
    logger.info("Processing %d bag_id(s): %s", len(bag_ids), bag_ids)

    sample = SimpleNamespace(
        slide_ids=[slide_id],
        artifact_paths=[artifact_path],
        metadata={"dataset": str(args.dataset)},
    )
    for bag_id in bag_ids:
        mean_rgb = resolve_sample_patch_mean_rgb(sample=sample, bag_id=bag_id, config=cfg)
        logger.info(
            "Resolved mean_rgb for bag_id='%s' with shape=%s",
            bag_id,
            tuple(mean_rgb.shape),
        )

    logger.info("mean_rgb precompute finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
