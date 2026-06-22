from __future__ import annotations

from pathlib import Path

import pytest

from pathbench.config.config import Config


@pytest.mark.smoke
def test_feature_extract_slide_cli_importable() -> None:
    from pathbench.cli import feature_extraction_slide  # noqa: F401


@pytest.mark.smoke
def test_feature_extract_slide_cli_missing_config(tmp_path: Path) -> None:
    from pathbench.cli.feature_extraction_slide import main

    with pytest.raises(FileNotFoundError):
        main(
            [
                "--config",
                str(tmp_path / "missing.yaml"),
                "--dataset",
                "ds",
                "--input",
                str(tmp_path / "slide.svs"),
            ]
        )


@pytest.mark.smoke
def test_feature_extract_slide_cli_missing_slide(tmp_path: Path) -> None:
    from pathbench.cli.feature_extraction_slide import main

    ann = tmp_path / "annotations.csv"
    ann.write_text("dataset,slide,patient,category\n", encoding="utf-8")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
experiment:
  project_name: smoke_slide
  annotation_file: {ann}
  project_root: {tmp_path / "project"}
  mode: feature_extraction
slide_processing:
  backend: lazyslide
datasets:
  - name: ds
    slides_dir: {tmp_path / "slides"}
    artifacts_dir: {tmp_path / "artifacts"}
    used_for: all
benchmark_parameters:
  tile_px: [256]
  tile_mpp: [0.5]
  feature_extraction: [dummy_fe]
  mil: []
""",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        main(
            [
                "--config",
                str(cfg_path),
                "--dataset",
                "ds",
                "--input",
                str(tmp_path / "nonexistent.svs"),
            ]
        )


@pytest.mark.smoke
def test_feature_extraction_config_validates(
    minimal_fe_config: dict[str, object],
) -> None:
    cfg = Config.model_validate(minimal_fe_config)
    assert cfg.experiment.mode == "feature_extraction"
    assert cfg.benchmark_parameters.tile_px == [256]
