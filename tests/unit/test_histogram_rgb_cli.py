from types import SimpleNamespace

import pytest

from pathforge.cli import retrieval_histogram_rgb as cli


@pytest.mark.parametrize('available', [True, False])
def test_histogram_cli_prepares_tiles_before_requesting_quality(tmp_path, monkeypatch, available):
    config_path = tmp_path / 'config.yaml'
    config_path.touch()
    artifact = tmp_path / 'slide.h5'
    if available:
        artifact.touch()
    cfg = SimpleNamespace(datasets=[SimpleNamespace(name='reference', artifacts_dir=tmp_path)])
    monkeypatch.setattr(cli.Config, 'from_yaml', lambda _: cfg)
    monkeypatch.setattr(cli, '_resolve_bag_ids', lambda *a: ['256px_0.5mpp'])
    calls = []
    prepared = []
    monkeypatch.setattr(cli, '_ensure_sample_tiles', lambda **k: prepared.append(k))

    def resolve(**kwargs):
        assert len(prepared) == 1
        calls.append(kwargs)

    monkeypatch.setattr(cli, 'resolve_sample_patch_rgb_descriptors', resolve)
    assert cli.run_histogram_rgb(config=config_path, dataset='reference', slide_id='slide') == 0
    assert len(calls) == 1
    assert calls[0]['include_quality'] is True
    assert calls[0]['bag_id'] == '256px_0.5mpp'
    assert calls[0]['sample'].artifact_paths == [artifact]
    assert calls[0]['config'] is cfg
