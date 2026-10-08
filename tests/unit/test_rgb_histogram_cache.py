from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pathforge.core.io.slide_artifacts import tiles as tiles_io
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.core.io.slide_retrieval import descriptors as descriptors_io
from pathforge.slide_retrieval.representation_strategies import histogram_rgb as rgb
from pathforge.slide_retrieval.representation_strategies.strategies import sish_rgb


@pytest.fixture()
def cached_slide(tmp_path):
    """Provide aligned tile rows and real H5 colour/quality descriptor storage."""
    artifact = tmp_path / 'slide.h5'
    coords = np.array([[0, 0, 256, 256, 0], [256, 0, 256, 256, 0],
                       [512, 0, 256, 256, 0]], dtype=np.int32)
    tile_id = '256px_0.5mpp'
    with FileHandleH5(artifact, mode='a') as handle:
        tiles_io.write_coords(handle, tile_id, coords)
    sample = SimpleNamespace(sample_id='slide', slide_ids=['slide'],
                             artifact_paths=[artifact], metadata={})
    target = rgb._slide_retrieval_artifact_path(slide_artifact_path=artifact, slide_id='slide')
    matrices = {'histogram_rgb': np.zeros((3, 768), dtype=np.float32),
                'sish_quality': np.zeros((3, 129), dtype=np.float32)}
    matrices['sish_quality'][:, 0] = [1, 2, 3]
    matrices['sish_quality'][:, 128] = [0, 1, 0]
    with FileHandleH5(target, mode='a') as handle:
        for name, contract in [('histogram_rgb', rgb._CONTRACT),
                               ('sish_quality', rgb._QUALITY_CONTRACT)]:
            descriptors_io.write_descriptor(handle, tile_id, name, matrices[name], metadata=contract)
    return sample, tile_id, coords, target, matrices


def test_complete_cache_loads_and_sish_selects_without_source_slide(cached_slide, monkeypatch):
    sample, tile_id, coords, _, expected = cached_slide
    def no_slide(**kwargs):
        raise AssertionError('Complete caches must not resolve or open a source WSI')
    monkeypatch.setattr(rgb, '_resolve_sample_slide_paths', no_slide)
    monkeypatch.setattr(rgb, '_build_slide_processor', no_slide)
    features = np.arange(12, dtype=np.float32).reshape(3, 4)
    monkeypatch.setattr(sish_rgb.features_io, 'read_features', lambda *a, **k: features)
    class Classifier:
        def predict(self, rows):
            # The white tile is excluded before classification.
            np.testing.assert_array_equal(rows[:, 0], [1, 3])
            return np.array([0, 1])
    monkeypatch.setattr(sish_rgb, 'load_sish_trash_classifier', lambda **k: Classifier())
    strategy = sish_rgb.SISHRGB(params={'n_clusters': 1}, config=SimpleNamespace())
    payload = strategy.load_sample(index=0, sample=sample,
                                  base_dataset=SimpleNamespace(tiling_id=tile_id, extractor_name='fm'))
    np.testing.assert_array_equal(payload['sish_quality'], expected['sish_quality'])
    result = strategy.run(sample=sample, **payload)
    np.testing.assert_array_equal(result.data, features[[0]])
    np.testing.assert_array_equal(result.additional_data['selected_coords'], coords[[0], :2])
    np.testing.assert_array_equal(result.additional_data['group_ids'], [0, -1, -1])


@pytest.mark.parametrize('invalid', ['missing', 'rows', 'metadata'])
def test_quality_cache_upgrade_preserves_valid_rgb(cached_slide, monkeypatch, tmp_path, invalid):
    sample, tile_id, _, target, expected = cached_slide
    with FileHandleH5(target, mode='a') as handle:
        path = descriptors_io.DEFAULT_LAYOUT.descriptor(tile_id, 'sish_quality')
        if invalid == 'missing':
            del handle.h5[path]
        elif invalid == 'rows':
            del handle.h5[path]
            descriptors_io.write_descriptor(handle, tile_id, 'sish_quality', np.zeros((1, 129)),
                                            metadata=rgb._QUALITY_CONTRACT)
        else:
            handle.h5[path].attrs['version'] = 0
    # Yottixel can continue using its valid RGB-only cache before the upgrade.
    np.testing.assert_array_equal(rgb.resolve_sample_patch_histogram_rgb(
        sample=sample, bag_id=tile_id, config=object()), expected['histogram_rgb'])
    slide = tmp_path / 'source.svs'
    slide.touch()
    monkeypatch.setattr(rgb, '_resolve_sample_slide_paths', lambda **k: {'slide': slide})
    monkeypatch.setattr(rgb, '_build_slide_processor', lambda **k: object())
    sample.metadata = {'dataset': 'reference'}
    sample.annotations_df = pd.DataFrame([
        {'dataset': 'reference', 'slide': 'slide', 'fallback_mpp': 0.25}
    ])
    def create(*args, **kwargs):
        assert kwargs['include_quality'] is True
        assert kwargs['annotation_row']['fallback_mpp'] == 0.25
        return {'histogram_rgb': np.ones((3, 768), dtype=np.float32),
                'sish_quality': expected['sish_quality']}
    monkeypatch.setattr(rgb, '_create_slide_descriptors', create)
    resolved = rgb.resolve_sample_patch_rgb_descriptors(
        sample=sample, bag_id=tile_id, config=object(), include_quality=True)
    for name, values in expected.items():
        np.testing.assert_array_equal(resolved[name], values)
        with FileHandleH5(target, mode='r') as handle:
            np.testing.assert_array_equal(descriptors_io.read_descriptor(handle, tile_id, name), values)


def test_missing_quality_without_slide_explains_required_descriptor(cached_slide):
    sample, tile_id, _, target, _ = cached_slide
    sample.slide_paths = [None]
    with FileHandleH5(target, mode='a') as handle:
        del handle.h5[descriptors_io.DEFAULT_LAYOUT.descriptor(tile_id, 'sish_quality')]
    with pytest.raises(FileNotFoundError, match='Missing stored sish_quality'):
        rgb.resolve_sample_patch_rgb_descriptors(sample=sample, bag_id=tile_id,
                                                config=object(), include_quality=True)


def test_empty_tiles_cache_without_slide(cached_slide, monkeypatch):
    sample, tile_id, _, _, _ = cached_slide
    with FileHandleH5(sample.artifact_paths[0], mode='a') as handle:
        tiles_io.write_coords(handle, tile_id, np.empty((0, 5), dtype=np.int32))
    monkeypatch.setattr(rgb, '_resolve_sample_slide_paths', lambda **k: pytest.fail('No WSI needed'))
    resolved = rgb.resolve_sample_patch_rgb_descriptors(
        sample=sample, bag_id=tile_id, config=object(), include_quality=True)
    assert resolved['histogram_rgb'].shape == (0, 768)
    assert resolved['sish_quality'].shape == (0, 129)


@pytest.mark.parametrize('sample', [None, SimpleNamespace(slide_ids=[], artifact_paths=[])])
def test_cache_rejects_missing_sample_rows(sample):
    with pytest.raises(ValueError):
        rgb.resolve_sample_patch_rgb_descriptors(sample=sample, bag_id='256px_0.5mpp', config=object())


@pytest.mark.parametrize("include_quality", [False, True])
@pytest.mark.parametrize("fallback_mpp", [None, 0.25])
def test_create_descriptors_reads_tiles_once_and_matches_direct_filter(
    cached_slide, tmp_path, include_quality, fallback_mpp
):
    import cv2 as cv
    sample, _, coords, _, _ = cached_slide
    patch = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    class Processor:
        reads = 0
        closed = False
        def load_wsi(self, wsi):
            assert wsi.fallback_mpp == fallback_mpp
        def read_patch_region(self, wsi, **kwargs):
            self.reads += 1
            return patch
        def close_wsi(self, wsi):
            self.closed = True
    processor = Processor()
    result = rgb._create_slide_descriptors('slide', tmp_path / 'source.svs',
                                          sample.artifact_paths[0], coords, processor,
                                          include_quality=include_quality,
                                          annotation_row={'slide': 'slide', 'fallback_mpp': fallback_mpp})
    if include_quality:
        from skimage import feature as skfeature

        grey = cv.resize(cv.cvtColor(patch, cv.COLOR_RGB2GRAY), (256, 256))
        expected_lbp = np.histogram(skfeature.local_binary_pattern(grey, 8, 1, 'ror'),
                                    density=True, bins=128, range=(0, 128))[0]
        np.testing.assert_allclose(result['sish_quality'][:, :128],
                                   np.tile(expected_lbp, (len(coords), 1)), rtol=1e-6)
        np.testing.assert_allclose(result['sish_quality'][:, 128], np.mean(grey > 235))
    else:
        assert 'sish_quality' not in result
    np.testing.assert_array_equal(result['histogram_rgb'].reshape(3, 3, 256).sum(axis=2),
                                  np.full((3, 3), 256 * 256))
    assert processor.reads == len(coords)
    assert processor.closed


@pytest.mark.parametrize("patch_shape", [(2, 2), (2, 2, 4)])
def test_create_descriptors_rejects_non_rgb_and_closes_slide(
    cached_slide, tmp_path, patch_shape
):
    sample, _, coords, _, _ = cached_slide

    class Processor:
        closed = False

        def load_wsi(self, wsi):
            pass

        def read_patch_region(self, wsi, **kwargs):
            return np.zeros(patch_shape, dtype=np.uint8)

        def close_wsi(self, wsi):
            self.closed = True

    processor = Processor()
    with pytest.raises(ValueError, match="Patch RGB reads must have shape"):
        rgb._create_slide_descriptors(
            "slide", tmp_path / "source.svs", sample.artifact_paths[0],
            coords, processor, include_quality=False,
        )
    assert processor.closed
