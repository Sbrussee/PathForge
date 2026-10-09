"""CLI for materializing RGB histograms and SISH quality descriptors."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import typer

from pathforge.cli.common import configure_logging, resolve_config_path
from pathforge.cli.retrieval_mean_rgb import (
    _ensure_sample_tiles,
    _find_dataset_config,
    _resolve_bag_ids,
)
from pathforge.config.config import Config
from pathforge.slide_retrieval.representation_strategies.histogram_rgb import (
    resolve_sample_patch_rgb_descriptors,
)

_CONFIG_OPTION = typer.Option(..., "--config")
_DATASET_OPTION = typer.Option(..., "--dataset")
_SLIDE_ID_OPTION = typer.Option(..., "--slide-id")
_INPUT_OPTION = typer.Option(None, "--input")
_BAG_ID_OPTION = typer.Option(None, "--bag-id")
_ARTIFACT_OPTION = typer.Option(None, "--artifact-path")
_LOG_OPTION = typer.Option("INFO", "--log-level")


def run_histogram_rgb(
    *,
    config: Path,
    dataset: str,
    slide_id: str,
    input_path: Path | None = None,
    bag_ids: list[str] | None = None,
    artifact_path: Path | None = None,
    log_level: str = "INFO",
) -> int:
    """Cache RGB ``(N,768)`` and quality ``(N,129)`` rows for one slide's tilings.

    Returns exit code zero after cache creation or validation. Example:
    ``run_histogram_rgb(config=Path("retrieval.yaml"), dataset="ReferenceSet",
    slide_id="slide-1")``. The source WSI is read for missing tiles or descriptors.
    """
    configure_logging(log_level)
    cfg = Config.from_yaml(resolve_config_path(config))
    dataset_cfg = _find_dataset_config(cfg, str(dataset))
    artifact = (
        Path(artifact_path).expanduser().resolve()
        if artifact_path
        else Path(dataset_cfg.artifacts_dir).expanduser().resolve() / f"{slide_id}.h5"
    )
    source = Path(input_path).expanduser().resolve() if input_path else None
    if source is not None and not source.is_file():
        raise FileNotFoundError(f"Input slide not found: {source}")
    sample = SimpleNamespace(
        slide_ids=[str(slide_id)],
        artifact_paths=[artifact],
        slide_paths=[source] if source else [],
        metadata={"dataset": str(dataset)},
    )
    for bag_id in _resolve_bag_ids(cfg, bag_ids):
        _ensure_sample_tiles(sample=sample, bag_id=bag_id, config=cfg, dataset_cfg=dataset_cfg)
        resolve_sample_patch_rgb_descriptors(
            sample=sample, bag_id=bag_id, config=cfg, include_quality=True
        )
    return 0


def run_command(
    config: Path = _CONFIG_OPTION,
    dataset: str = _DATASET_OPTION,
    slide_id: str = _SLIDE_ID_OPTION,
    input_path: Path | None = _INPUT_OPTION,
    bag_ids: list[str] | None = _BAG_ID_OPTION,
    artifact_path: Path | None = _ARTIFACT_OPTION,
    log_level: str = _LOG_OPTION,
) -> None:
    """Precompute RGB, LBP, and white-fraction caches for one slide."""
    raise SystemExit(
        run_histogram_rgb(
            config=config,
            dataset=dataset,
            slide_id=slide_id,
            input_path=input_path,
            bag_ids=bag_ids,
            artifact_path=artifact_path,
            log_level=log_level,
        )
    )
