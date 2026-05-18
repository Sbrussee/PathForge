from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import typer

from pathbench.cli.common import LOG_LEVEL_CHOICES, configure_logging, resolve_config_path
from pathbench.cli.retrieval_mean_rgb import _find_dataset_config, _resolve_bag_ids
from pathbench.config.config import Config
from pathbench.slide_retrieval.search_strategies.strategies.sish.sish_vqvae_descriptors import (
    SISH_VQVAE_DESCRIPTOR_NAME,
    resolve_sample_patch_sish_vqvae_latent,
)

logger = logging.getLogger(__name__)


def run_sish_vqvae(
    *,
    config: Path,
    dataset: str,
    slide_id: str,
    input_path: Path | None = None,
    bag_ids: list[str] | None = None,
    artifact_path: Path | None = None,
    descriptor_name: str = SISH_VQVAE_DESCRIPTOR_NAME,
    log_level: str = "INFO",
) -> int:
    configure_logging(log_level)
    config_path = resolve_config_path(config)

    cfg = Config.from_yaml(config_path)
    dataset_cfg = _find_dataset_config(cfg, dataset_name=str(dataset))
    normalized_slide_id = str(slide_id).strip()
    if not normalized_slide_id:
        raise ValueError("--slide-id must be a non-empty string.")

    input_slide_path = None
    if input_path is not None:
        input_slide_path = Path(input_path).expanduser().resolve()
        if not input_slide_path.is_file():
            raise FileNotFoundError(f"Input slide not found: {input_slide_path}")

    resolved_artifact_path = (
        Path(artifact_path).expanduser().resolve()
        if artifact_path is not None
        else Path(dataset_cfg.artifacts_dir).expanduser().resolve() / f"{normalized_slide_id}.h5"
    )
    if not resolved_artifact_path.is_file():
        raise FileNotFoundError(
            f"Slide artifact file not found for slide_id='{normalized_slide_id}': {resolved_artifact_path}"
        )

    resolved_bag_ids = _resolve_bag_ids(cfg, explicit_bag_ids=bag_ids)
    if not resolved_bag_ids:
        raise ValueError(
            "No bag IDs provided and none could be inferred from config.benchmark_parameters "
            "(tile_px/tile_mpp)."
        )

    logger.info("Starting SISH VQ-VAE descriptor CLI")
    logger.info("Using config: %s", config_path)
    logger.info("Using dataset: %s", dataset)
    logger.info("Using slide_id: %s", normalized_slide_id)
    if input_slide_path is not None:
        logger.info("Using input slide: %s", input_slide_path)
    logger.info("Using artifact: %s", resolved_artifact_path)
    logger.info("Using descriptor name: %s", descriptor_name)
    logger.info("Processing %d bag_id(s): %s", len(resolved_bag_ids), resolved_bag_ids)

    sample = SimpleNamespace(
        slide_ids=[normalized_slide_id],
        artifact_paths=[resolved_artifact_path],
        slide_paths=([input_slide_path] if input_slide_path is not None else []),
        metadata={"dataset": str(dataset)},
    )
    for bag_id in resolved_bag_ids:
        descriptor_matrix = resolve_sample_patch_sish_vqvae_latent(
            sample=sample,
            bag_id=bag_id,
            config=cfg,
            descriptor_name=descriptor_name,
        )
        logger.info(
            "Resolved SISH VQ-VAE descriptor for bag_id='%s' with shape=%s",
            bag_id,
            tuple(descriptor_matrix.shape),
        )

    logger.info("SISH VQ-VAE descriptor precompute finished.")
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
        help="Optional explicit source slide path.",
    ),
    bag_ids: list[str] | None = typer.Option(
        None,
        "--bag-id",
        help="Bag identifier (repeatable).",
    ),
    artifact_path: Path | None = typer.Option(
        None,
        "--artifact-path",
        help="Optional explicit slide artifact path.",
    ),
    descriptor_name: str = typer.Option(
        SISH_VQVAE_DESCRIPTOR_NAME,
        "--descriptor-name",
        help="Retrieval descriptor dataset name to write.",
        show_default=True,
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level.",
        show_default=True,
    ),
) -> None:
    raise SystemExit(
        run_sish_vqvae(
            config=config,
            dataset=dataset,
            slide_id=slide_id,
            input_path=input_path,
            bag_ids=bag_ids,
            artifact_path=artifact_path,
            descriptor_name=descriptor_name,
            log_level=log_level,
        )
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute and persist SISH VQ-VAE patch descriptors for one slide."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset name")
    parser.add_argument("--slide-id", required=True, type=str, help="Slide ID")
    parser.add_argument("--input", type=Path, default=None, help="Optional explicit source slide path.")
    parser.add_argument("--bag-id", action="append", default=None, help="Bag identifier (repeatable).")
    parser.add_argument("--artifact-path", type=Path, default=None, help="Optional explicit slide artifact path.")
    parser.add_argument(
        "--descriptor-name",
        default=SISH_VQVAE_DESCRIPTOR_NAME,
        type=str,
        help="Retrieval descriptor dataset name to write.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVEL_CHOICES,
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args(argv)
    return run_sish_vqvae(
        config=args.config,
        dataset=args.dataset,
        slide_id=args.slide_id,
        input_path=args.input,
        bag_ids=args.bag_id,
        artifact_path=args.artifact_path,
        descriptor_name=args.descriptor_name,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    raise SystemExit(main())
