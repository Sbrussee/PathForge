"""Regression coverage for RGB preprocessing without pre-existing tile artifacts."""

from types import SimpleNamespace

import numpy as np
import pytest

from pathforge.cli import retrieval_histogram_rgb, retrieval_mean_rgb
from pathforge.core.datasets.wsi_dataset import WSI
from pathforge.core.experiments.combinations import ComboConfig
from pathforge.core.io.slide_artifacts import tiles as tiles_io
from pathforge.core.io.slide_artifacts import tissue as tissue_io
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.core.io.slide_retrieval import descriptors as descriptors_io
from pathforge.policy.feature_extraction import FeatureExtractionPolicy
from pathforge.slide_retrieval.representation_strategies import histogram_rgb, mean_rgb

TILING_ID = "256px_0.5mpp"
SPEC = {"tile_px": 256, "tile_mpp": 0.5, "stride_px": 256, "coord_space": "level0"}
POLYGONS = [[[[0, 0], [512, 0], [512, 512], [0, 0]]]]


class TileProcessor:
    """Deterministic source backend: one constant RGB patch, no feature model."""

    def __init__(self, coords):
        self.coords = coords
        self.loaded = []
        self.closed = []
        self.segmented = 0
        self.extracted = 0
        self.released = 0

    def load_wsi(self, wsi):
        self.loaded.append(wsi)

    def get_base_mpp(self, wsi):
        return 0.5

    def segment_tissue(self, wsi, *, config):
        self.segmented += 1
        return POLYGONS

    def extract_patches(self, wsi, polygons, *, config):
        assert polygons == POLYGONS
        self.extracted += 1
        return self.coords, dict(SPEC)

    def read_patch_region(self, wsi, **kwargs):
        return np.full((4, 4, 3), [64, 128, 192], dtype=np.uint8)

    def close_wsi(self, wsi):
        self.closed.append(wsi)

    def close(self):
        self.released += 1


@pytest.fixture()
def context(tmp_path, monkeypatch):
    ds = SimpleNamespace(
        name="reference", artifacts_dir=tmp_path / "artifacts",
        slides_dir=tmp_path, tissue_annotations_dir=None,
    )
    cfg = SimpleNamespace(
        datasets=[ds],
        slide_processing=SimpleNamespace(backend="test", segmentation_method="test", qc_filters=[]),
    )
    source = tmp_path / "slide.svs"
    source.touch()
    artifact = ds.artifacts_dir / "slide.h5"
    wsi = WSI.from_annotation(
        {"slide": "slide", "fallback_mpp": 0.5}, slide_path=source, artifact_path=artifact
    )
    processor = TileProcessor(np.array([[0, 0, 4, 4, 0]], dtype=np.int32))
    monkeypatch.setattr(FeatureExtractionPolicy, "_build_processor", lambda self: processor)
    monkeypatch.setattr(mean_rgb, "_build_slide_processor", lambda **kwargs: processor)
    monkeypatch.setattr(histogram_rgb, "_build_slide_processor", lambda **kwargs: processor)
    policy = FeatureExtractionPolicy(SimpleNamespace(cfg=cfg))
    return cfg, ds, wsi, processor, policy


@pytest.mark.parametrize("empty", [False, True])
def test_ensure_tiles_persists_and_returns_tiles_without_features(context, empty):
    cfg, ds, wsi, processor, policy = context
    if empty:
        processor.coords = np.empty((0, 5), dtype=np.int32)
    coords, spec = policy.ensure_tiles(ds, wsi, ComboConfig(tile_px=256, tile_mpp=0.5))
    np.testing.assert_array_equal(coords, processor.coords)
    assert spec == SPEC
    with FileHandleH5(wsi.artifact_path, mode="r") as artifact:
        np.testing.assert_array_equal(tiles_io.read_coords(artifact, TILING_ID), coords)
        assert tiles_io.read_tiling_spec(artifact, TILING_ID) == SPEC
        assert tissue_io.read_tissue(artifact) == POLYGONS
        assert "features" not in artifact.h5[f"bags/{TILING_ID}"]
    assert processor.segmented == processor.extracted == 1
    assert processor.closed == [wsi]
    assert processor.released == 1

    # Cached coordinates can be returned after the source slide disappears.
    wsi.path.unlink()
    loaded, loaded_spec = policy.ensure_tiles(ds, wsi, ComboConfig(tile_px=256, tile_mpp=0.5))
    np.testing.assert_array_equal(loaded, coords)
    assert loaded_spec == spec
    assert processor.loaded == [wsi]


def test_ensure_tiles_reuses_tissue_and_preserves_other_tilings(context):
    _, ds, wsi, processor, policy = context
    wsi.artifact_path.parent.mkdir()
    with FileHandleH5(wsi.artifact_path, mode="a") as artifact:
        tissue_io.write_tissue(artifact, POLYGONS)
        tiles_io.write_coords(artifact, "128px_0.5mpp", processor.coords)
    policy.ensure_tiles(ds, wsi, ComboConfig(tile_px=256, tile_mpp=0.5))
    assert processor.segmented == 0
    with FileHandleH5(wsi.artifact_path, mode="r") as artifact:
        assert tiles_io.coords_exist(artifact, "128px_0.5mpp")
        assert tiles_io.coords_exist(artifact, TILING_ID)


def test_ensure_tiles_closes_processor_on_extraction_failure(context, monkeypatch):
    _, ds, wsi, processor, policy = context
    def fail(*args, **kwargs):
        raise RuntimeError("tiling failed")
    monkeypatch.setattr(processor, "extract_patches", fail)
    with pytest.raises(RuntimeError, match="tiling failed"):
        policy.ensure_tiles(ds, wsi, ComboConfig(tile_px=256, tile_mpp=0.5))
    assert processor.closed == [wsi]
    assert processor.released == 1
    assert not wsi.artifact_path.exists()


@pytest.mark.parametrize("cli", [retrieval_mean_rgb, retrieval_histogram_rgb])
@pytest.mark.parametrize("artifact_exists", [False, True])
def test_rgb_cli_creates_tiles_before_descriptors(context, tmp_path, monkeypatch, cli, artifact_exists):
    cfg, ds, wsi, processor, _ = context
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    monkeypatch.setattr(cli.Config, "from_yaml", lambda _: cfg)
    if cli is retrieval_histogram_rgb:
        create = histogram_rgb._create_slide_descriptors

        def create_with_quality(*args, include_quality, **kwargs):
            # Exercise actual colour creation and quality persistence/dispatch;
            # the unchanged LBP calculation has its own descriptor tests.
            assert include_quality is True
            matrices = create(*args, include_quality=False, **kwargs)
            matrices["sish_quality"] = np.zeros((len(args[3]), 129), dtype=np.float32)
            return matrices

        monkeypatch.setattr(histogram_rgb, "_create_slide_descriptors", create_with_quality)
    if artifact_exists:
        wsi.artifact_path.parent.mkdir()
        with FileHandleH5(wsi.artifact_path, mode="a") as artifact:
            artifact.h5.create_dataset("unrelated", data=[42])
    run = cli.run_mean_rgb if cli is retrieval_mean_rgb else cli.run_histogram_rgb
    assert run(
        config=config_path, dataset=ds.name, slide_id=wsi.slide,
        input_path=wsi.path, bag_ids=[TILING_ID],
    ) == 0
    with FileHandleH5(wsi.artifact_path, mode="r") as artifact:
        np.testing.assert_array_equal(tiles_io.read_coords(artifact, TILING_ID), processor.coords)
        if artifact_exists:
            assert artifact.h5["unrelated"][0] == 42
    target = mean_rgb._slide_retrieval_artifact_path(
        slide_artifact_path=wsi.artifact_path, slide_id=wsi.slide
    )
    with FileHandleH5(target, mode="r") as artifact:
        if cli is retrieval_mean_rgb:
            matrix = descriptors_io.read_descriptor(artifact, TILING_ID, "mean_rgb")
            np.testing.assert_allclose(matrix, [[64 / 255, 128 / 255, 192 / 255]])
        else:
            matrix = descriptors_io.read_descriptor(artifact, TILING_ID, "histogram_rgb")
            assert matrix.shape == (1, 768)
            np.testing.assert_array_equal(matrix[0, [64, 256 + 128, 512 + 192]], [65536] * 3)
            quality = descriptors_io.read_descriptor(artifact, TILING_ID, "sish_quality")
            assert quality.shape == (1, 129)
            assert quality[0, -1] == 0
    # All descriptors and tiles are now cached; no source or annotation reads are needed.
    wsi.path.unlink()
    loaded_count = len(processor.loaded)
    assert run(config=config_path, dataset=ds.name, slide_id=wsi.slide, bag_ids=[TILING_ID]) == 0
    assert len(processor.loaded) == loaded_count


@pytest.mark.parametrize("bag_id", ["invalid", "0px_0.5mpp", "256px_-1mpp", "256px_0.5mpp__model"])
def test_cli_tile_preparation_rejects_invalid_tiling_ids(context, bag_id):
    cfg, ds, wsi, _, _ = context
    sample = SimpleNamespace(slide_ids=[wsi.slide], artifact_paths=[wsi.artifact_path])
    with pytest.raises(ValueError, match="Invalid canonical tiling ID"):
        retrieval_mean_rgb._ensure_sample_tiles(sample=sample, bag_id=bag_id, config=cfg, dataset_cfg=ds)


def test_cli_tile_preparation_reports_missing_source(context):
    cfg, ds, wsi, processor, _ = context
    wsi.path.unlink()
    sample = SimpleNamespace(
        slide_ids=[wsi.slide], artifact_paths=[wsi.artifact_path], metadata={"dataset": ds.name}
    )
    with pytest.raises(FileNotFoundError, match="Missing tiles and no source slide"):
        retrieval_mean_rgb._ensure_sample_tiles(sample=sample, bag_id=TILING_ID, config=cfg, dataset_cfg=ds)
    assert not processor.loaded
