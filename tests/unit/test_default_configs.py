"""Schema and catalog checks for the shipped default configuration templates."""

from __future__ import annotations

from pathlib import Path

import pytest

from pathforge.config.config import Config


DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "default_config"
EXPECTED_CONFIGS = {
    "benchmark_classification.yaml",
    "benchmark_regression.yaml",
    "benchmark_slide_retrieval.yaml",
    "benchmark_survival_continuous.yaml",
    "benchmark_survival_discrete.yaml",
    "feature_extraction_only.yaml",
    "optimize_classification.yaml",
    "optimize_regression.yaml",
    "optimize_slide_retrieval.yaml",
    "optimize_survival_continuous.yaml",
    "optimize_survival_discrete.yaml",
    "parallel_benchmark.yaml",
    "parallel_optimization.yaml",
}


@pytest.mark.parametrize("filename", sorted(EXPECTED_CONFIGS))
def test_default_config_validates_against_live_schema(filename: str) -> None:
    """Every shipped template must resolve against current registries and fields."""

    config = Config.from_yaml(DEFAULT_CONFIG_DIR / filename)

    assert config.benchmark_parameters.tile_px == [224]
    assert config.benchmark_parameters.get_values("feature_extraction") == [
        "h-optimus-1",
        "uni2",
        "virchow2",
        "gpfm",
    ]


def test_default_config_catalog_is_complete() -> None:
    """The folder contains exactly the documented runnable YAML templates."""

    actual = {path.name for path in DEFAULT_CONFIG_DIR.glob("*.yaml")}
    assert actual == EXPECTED_CONFIGS


@pytest.mark.parametrize(
    "filename",
    [
        "benchmark_classification.yaml",
        "benchmark_regression.yaml",
        "benchmark_survival_continuous.yaml",
        "benchmark_survival_discrete.yaml",
    ],
)
def test_mil_defaults_cover_requested_pipeline_grid(filename: str) -> None:
    """MIL templates retain the requested models, magnifications, and normalization."""

    config = Config.from_yaml(DEFAULT_CONFIG_DIR / filename)

    assert config.benchmark_parameters.get_values("mil") == [
        "ABMIL",
        "DSMIL",
        "TransMIL",
    ]
    assert config.benchmark_parameters.tile_mpp == [0.5, 1.0]
    assert config.benchmark_parameters.get_values("color_norm") == [None, "macenko"]
    assert len(config.benchmark_parameters.loss) == 2
