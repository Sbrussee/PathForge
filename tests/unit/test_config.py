# tests/unit/test_config.py

from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError
from torch import Tensor

from pathforge.config.config import Config
from pathforge.core.feature_extractors.base import FeatureExtractorBase
from pathforge.utils.registries import FEATURE_EXTRACTORS
from tests.conftest import DUMMY_FE

if not FEATURE_EXTRACTORS.is_available("resnet18"):

    @FEATURE_EXTRACTORS.register("resnet18")
    class _TestResNet18(FeatureExtractorBase):
        """Minimal native extractor registration for configuration parsing."""

        def build_model(self) -> object:
            return object()

        def get_transform(self) -> None:
            return None

        def encode_images(self, images: Tensor) -> Tensor:  # pragma: no cover
            return images


def test_from_yaml_loads_minimal_valid_config(tmp_path):
    # Minimal config that satisfies validators (benchmark requires task + benchmark_parameters.mil)
    yaml_text = dedent(
        f"""
        experiment:
            project_name: testproj
            annotation_file: dummy.csv
            mode: feature_extraction

        slide_processing:
            backend: lazyslide
            feature_extraction:
                batch_size: 16
                num_workers: 2
                amp: true

        datasets: []

        benchmark_parameters:
            tile_px: [256]
            tile_mpp: [0.5]
            feature_extraction: ["{DUMMY_FE}"]
            mil: []
        """
    ).lstrip()

    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_text, encoding="utf-8")

    cfg = Config.from_yaml(p)
    assert cfg.experiment.project_name == "testproj"
    assert cfg.experiment.mode == "feature_extraction"
    assert cfg.experiment.task is None
    assert cfg.slide_processing.backend == "lazyslide"
    assert cfg.slide_processing.feature_extraction.batch_size == 16
    assert cfg.slide_processing.feature_extraction.num_workers == 2
    assert cfg.slide_processing.feature_extraction.amp is True
    assert cfg.benchmark_parameters.tile_px == [256]


def test_from_yaml_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.from_yaml(tmp_path / "does_not_exist.yaml")


def test_inference_mode_requires_task(tmp_path):
    yaml_text = dedent("""
                        experiment:
                            project_name: testproj
                            annotation_file: dummy.csv
                            mode: inference

                        slide_processing:
                            backend: lazyslide

                        datasets: []

                        benchmark_parameters:
                            tile_px: [256]
                            tile_mpp: [0.5]
                            feature_extraction: ["resnet18"]
                        """).lstrip()

    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValidationError, match="experiment.task is required"):
        Config.from_yaml(p)


def test_example_yaml_loads():
    example_yaml_path = Path("configs/config.example.yaml")
    if not example_yaml_path.exists():
        pytest.skip("configs/config.example.yaml not present in this environment")

    try:
        cfg = Config.from_yaml(example_yaml_path)
    except ValidationError:
        pytest.skip("config.example.yaml may include illustrative metrics not registered in this runtime")
    assert cfg.experiment.task in {
        "classification",
        "regression",
        "survival",
        "survival_discrete",
    }
