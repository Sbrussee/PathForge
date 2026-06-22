from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pathbench.cli.mean_rgb as mean_rgb_cli


def _fake_cfg(*, dataset_name: str, artifacts_dir: Path) -> SimpleNamespace:
    class _BenchmarkParams:
        def get_values(self, field_name: str) -> list[object]:
            if field_name == "tile_px":
                return [256]
            if field_name == "tile_mpp":
                return [0.5, 1.0]
            raise AssertionError(f"Unexpected benchmark parameter field: {field_name}")

    return SimpleNamespace(
        datasets=[
            SimpleNamespace(
                name=dataset_name,
                artifacts_dir=str(artifacts_dir),
            )
        ],
        benchmark_parameters=_BenchmarkParams(),
    )


def test_main_uses_explicit_bag_ids_and_artifact_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("fake: config", encoding="utf-8")

    artifact_path = tmp_path / "custom_artifact.h5"
    artifact_path.write_bytes(b"h5-placeholder")

    cfg = _fake_cfg(dataset_name="dataset-a", artifacts_dir=tmp_path / "unused")
    monkeypatch.setattr(mean_rgb_cli, "load_config", lambda _: cfg)

    calls: list[tuple[str, str, Path]] = []

    def _fake_resolve(*, sample, bag_id, config):
        assert config is cfg
        calls.append((str(sample.metadata["dataset"]), str(bag_id), Path(sample.artifact_paths[0])))
        return np.empty((1, 3), dtype=np.float32)

    monkeypatch.setattr(mean_rgb_cli, "resolve_sample_patch_mean_rgb", _fake_resolve)

    exit_code = mean_rgb_cli.main(
        [
            "--config",
            str(config_path),
            "--dataset",
            "dataset-a",
            "--slide-id",
            "slide-1",
            "--artifact-path",
            str(artifact_path),
            "--bag-id",
            "256px_0.5mpp",
            "--bag-id",
            "256px_0.5mpp",
            "--bag-id",
            "256px_1mpp",
        ]
    )

    assert exit_code == 0
    assert calls == [
        ("dataset-a", "256px_0.5mpp", artifact_path),
        ("dataset-a", "256px_1mpp", artifact_path),
    ]


def test_main_infers_bag_ids_from_config_when_not_provided(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("fake: config", encoding="utf-8")

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifacts_dir / "slide-1.h5"
    artifact_path.write_bytes(b"h5-placeholder")

    cfg = _fake_cfg(dataset_name="dataset-a", artifacts_dir=artifacts_dir)
    monkeypatch.setattr(mean_rgb_cli, "load_config", lambda _: cfg)

    seen_bag_ids: list[str] = []

    def _fake_resolve(*, sample, bag_id, config):
        assert config is cfg
        assert Path(sample.artifact_paths[0]) == artifact_path
        seen_bag_ids.append(str(bag_id))
        return np.empty((1, 3), dtype=np.float32)

    monkeypatch.setattr(mean_rgb_cli, "resolve_sample_patch_mean_rgb", _fake_resolve)

    exit_code = mean_rgb_cli.main(
        [
            "--config",
            str(config_path),
            "--dataset",
            "dataset-a",
            "--slide-id",
            "slide-1",
        ]
    )

    assert exit_code == 0
    assert seen_bag_ids == ["256px_0.5mpp", "256px_1mpp"]


def test_main_raises_for_unknown_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("fake: config", encoding="utf-8")

    cfg = _fake_cfg(dataset_name="dataset-a", artifacts_dir=tmp_path / "artifacts")
    monkeypatch.setattr(mean_rgb_cli, "load_config", lambda _: cfg)

    with pytest.raises(ValueError, match="Dataset 'dataset-b' not found in config.datasets"):
        mean_rgb_cli.main(
            [
                "--config",
                str(config_path),
                "--dataset",
                "dataset-b",
                "--slide-id",
                "slide-1",
                "--bag-id",
                "256px_0.5mpp",
            ]
        )
