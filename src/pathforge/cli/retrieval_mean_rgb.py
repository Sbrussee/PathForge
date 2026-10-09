from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import typer

from pathforge.cli.common import LOG_LEVEL_CHOICES, configure_logging, resolve_config_path
from pathforge.config.config import Config, DatasetEntry
from pathforge.core.datasets.wsi_dataset import WSI
from pathforge.core.experiments.combo_ids import build_tiling_id
from pathforge.core.experiments.combinations import ComboConfig
from pathforge.core.io.slide_artifacts import tiles as tiles_io
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.policy.feature_extraction import FeatureExtractionPolicy
from pathforge.slide_retrieval.annotations import resolve_sample_annotations
from pathforge.slide_retrieval.representation_strategies.mean_rgb import (
    _resolve_sample_slide_paths,
    resolve_sample_patch_mean_rgb,
)


logger = logging.getLogger(__name__)


def _ensure_sample_tiles(
    *, sample: Any, bag_id: str, config: Config, dataset_cfg: DatasetEntry
) -> None:
    """Ensure a single slide's canonical tiling is persisted before RGB reads.

    The policy returns coordinates shaped ``(N,5)`` and a specification; this
    CLI discards them because descriptor resolvers read the persisted artifact.
    Example: ``_ensure_sample_tiles(sample=s, bag_id="256px_0.5mpp",
    config=cfg, dataset_cfg=ds)``.
    """
    try:
        pixels, resolution = bag_id.split("px_")
        combo = ComboConfig(
            tile_px=int(pixels), tile_mpp=float(resolution.removesuffix("mpp"))
        )
        if combo.tile_px <= 0 or combo.tile_mpp <= 0 or build_tiling_id(combo) != bag_id:
            raise ValueError
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid canonical tiling ID: '{bag_id}'.") from error

    slide_id = sample.slide_ids[0]
    artifact = Path(sample.artifact_paths[0])
    # Cached runs should not require source discovery or annotation metadata.
    cached = False
    if artifact.is_file():
        with FileHandleH5(artifact, mode="r") as handle:
            cached = tiles_io.coords_exist(handle, bag_id) and tiles_io.tiling_spec_matches(
                handle,
                bag_id,
                {
                    "tile_px": combo.tile_px,
                    "tile_mpp": combo.tile_mpp,
                    "stride_px": combo.tile_px,
                    "coord_space": "level0",
                },
            )
    source = Path(dataset_cfg.slides_dir).expanduser().resolve() / slide_id
    row = {"slide": slide_id}
    if not cached:
        source = _resolve_sample_slide_paths(sample=sample, config=config)[slide_id]
        if source is None or not source.is_file():
            raise FileNotFoundError(
                f"Missing tiles and no source slide is available for slide '{slide_id}' "
                f"at tiling_id='{bag_id}'."
            )
        row = resolve_sample_annotations(sample=sample, config=config)[slide_id]
    wsi = WSI.from_annotation(row, slide_path=source, artifact_path=artifact)
    roi_root = dataset_cfg.tissue_annotations_dir
    dataset = SimpleNamespace(
        tissue_annotations_dir=Path(roi_root).expanduser().resolve() if roi_root else None
    )
    policy = FeatureExtractionPolicy(SimpleNamespace(cfg=config))
    policy.ensure_tiles(dataset=dataset, wsi=wsi, combo_cfg=combo)


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


def run_mean_rgb(
    *,
    config: Path,
    dataset: str,
    slide_id: str,
    input_path: Path | None = None,
    bag_ids: list[str] | None = None,
    artifact_path: Path | None = None,
    log_level: str = "INFO",
) -> int:
    """Ensure tiles exist, then precompute mean-RGB descriptors for one slide."""
    configure_logging(log_level)

    config_path = resolve_config_path(config)

    cfg = Config.from_yaml(config_path)
    dataset_cfg = _find_dataset_config(cfg, dataset_name=str(dataset))
    slide_id = str(slide_id).strip()
    if not slide_id:
        raise ValueError("--slide-id must be a non-empty string.")

    input_slide_path = None
    if input_path is not None:
        input_slide_path = Path(input_path).expanduser().resolve()
        if not input_slide_path.is_file():
            raise FileNotFoundError(f"Input slide not found: {input_slide_path}")

    artifact_path = (
        Path(artifact_path).expanduser().resolve()
        if artifact_path is not None
        else Path(dataset_cfg.artifacts_dir).expanduser().resolve() / f"{slide_id}.h5"
    )

    bag_ids = _resolve_bag_ids(cfg, explicit_bag_ids=bag_ids)
    if not bag_ids:
        raise ValueError(
            "No bag IDs provided and none could be inferred from config.benchmark_parameters "
            "(tile_px/tile_mpp)."
        )

    logger.info("Starting mean_rgb CLI")
    logger.info("Using config: %s", config_path)
    logger.info("Using dataset: %s", dataset)
    logger.info("Using slide_id: %s", slide_id)
    if input_slide_path is not None:
        logger.info("Using input slide: %s", input_slide_path)
    logger.info("Using artifact: %s", artifact_path)
    logger.info("Processing %d bag_id(s): %s", len(bag_ids), bag_ids)

    sample = SimpleNamespace(
        slide_ids=[slide_id],
        artifact_paths=[artifact_path],
        slide_paths=([input_slide_path] if input_slide_path is not None else []),
        metadata={"dataset": str(dataset)},
    )
    for bag_id in bag_ids:
        _ensure_sample_tiles(sample=sample, bag_id=bag_id, config=cfg, dataset_cfg=dataset_cfg)
        mean_rgb = resolve_sample_patch_mean_rgb(sample=sample, bag_id=bag_id, config=cfg)
        logger.info(
            "Resolved mean_rgb for bag_id='%s' with shape=%s",
            bag_id,
            tuple(mean_rgb.shape),
        )

    logger.info("mean_rgb precompute finished.")
    return 0


def run_command(
    config: Path = typer.Option(..., "--config", help="Path to YAML config"),
    dataset: str = typer.Option(
        ...,
        "--dataset",
        help="Dataset name (must exist in config.datasets).",
    ),
    slide_id: str = typer.Option(
        ...,
        "--slide-id",
        help="Slide ID (without extension), used for source lookup and artifact naming.",
    ),
    input_path: Path | None = typer.Option(
        None,
        "--input",
        help=(
            "Optional explicit source slide path. When provided, mean_rgb uses this "
            "path instead of resolving the slide from datasets[].slides_dir."
        ),
    ),
    bag_ids: list[str] | None = typer.Option(
        None,
        "--bag-id",
        help=(
            "Bag identifier (repeatable). If omitted, bag IDs are inferred from "
            "all tile_px/tile_mpp combinations in config.benchmark_parameters."
        ),
    ),
    artifact_path: Path | None = typer.Option(
        None,
        "--artifact-path",
        help=(
            "Optional explicit slide artifact path. Defaults to "
            "datasets[].artifacts_dir/{slide_id}.h5."
        ),
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level.",
        show_default=True,
    ),
) -> None:
    """Typer command that precomputes mean-RGB descriptors from the provided options."""
    raise SystemExit(
        run_mean_rgb(
            config=config,
            dataset=dataset,
            slide_id=slide_id,
            input_path=input_path,
            bag_ids=bag_ids,
            artifact_path=artifact_path,
            log_level=log_level,
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Argparse entry point for the mean-RGB CLI; returns a process exit code."""
    import argparse

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
        "--input",
        type=Path,
        default=None,
        help=(
            "Optional explicit source slide path. When provided, mean_rgb uses this "
            "path instead of resolving the slide from datasets[].slides_dir."
        ),
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
        choices=LOG_LEVEL_CHOICES,
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args(argv)
    return run_mean_rgb(
        config=args.config,
        dataset=args.dataset,
        slide_id=args.slide_id,
        input_path=args.input,
        bag_ids=args.bag_id,
        artifact_path=args.artifact_path,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    raise SystemExit(main())
