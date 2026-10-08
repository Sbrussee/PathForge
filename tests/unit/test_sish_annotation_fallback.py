from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from pathforge.core.io.slide_artifacts import tiles as tiles_io
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.core.slide_processing.lazyslide import LazySlideProcessor
from pathforge.slide_retrieval.representation_strategies.strategies import sish_rgb
from pathforge.slide_retrieval.search_strategies.strategies.sish import sish_precompute, sish_vqvae_descriptors


def test_sish_annotation_fallback_reaches_all_three_source_read_paths(tmp_path, monkeypatch):
    staged = tmp_path / "renamed.svs"
    staged.write_bytes(b"fake")
    artifact_path = tmp_path / "S1.h5"
    bag_id = "256px_0.5mpp"
    coords = np.array([[0, 0, 256, 256, 0]], dtype=np.int32)
    with FileHandleH5(artifact_path, mode="a") as artifact:
        tiles_io.write_coords(artifact, bag_id, coords)
        tiles_io.write_tiling_spec(artifact, bag_id, {"tile_px": 256, "tile_mpp": 0.5, "stride_px": 256, "coord_space": "level0"})
    frame = pd.DataFrame([{"dataset": "a", "slide": "S1", "fallback_mpp": 0.25}])
    sample = SimpleNamespace(sample_id="S1", slide_ids=["S1"], artifact_paths=[artifact_path], slide_paths=[staged], metadata={"dataset": "a"}, annotations_df=frame)
    cfg = SimpleNamespace()
    closed = []
    opened = []

    def open_slide(path, **kwargs):
        assert path == staged.resolve()
        obj = SimpleNamespace(properties=SimpleNamespace(mpp=None), close=lambda: closed.append(True))
        opened.append(obj)
        return obj

    monkeypatch.setattr("pathforge.core.slide_processing.lazyslide.processor.open_wsi", open_slide)
    processor = LazySlideProcessor()

    def crop(**kwargs):
        assert processor.get_base_mpp(kwargs["wsi"]) == 0.25
        return np.full((2, 2, 3), 128, dtype=np.uint8)

    monkeypatch.setattr(sish_vqvae_descriptors, "_build_slide_processor", lambda **kwargs: processor)
    monkeypatch.setattr(sish_vqvae_descriptors, "_load_sish_vqvae_encoder", lambda **kwargs: torch.nn.Identity())
    monkeypatch.setattr(sish_vqvae_descriptors, "read_canonical_sish_crop", crop)
    monkeypatch.setattr(sish_vqvae_descriptors, "_encode_latent_batch", lambda tensors, model: [np.array([1], dtype=np.float32) for _ in tensors])
    latents = sish_vqvae_descriptors.resolve_sample_patch_sish_vqvae_latent(sample=sample, bag_id=bag_id, config=cfg)
    np.testing.assert_array_equal(latents, [[1]])

    precompute = sish_precompute.SISHPrecompute(config=cfg)
    precompute._slide_processor = processor
    precompute._vqvae = torch.nn.Identity()
    monkeypatch.setattr(precompute, "_load_models", lambda: None)
    monkeypatch.setattr(sish_precompute, "read_canonical_sish_crop", crop)
    monkeypatch.setattr(sish_precompute, "_slide_to_index", lambda *args, **kwargs: np.array([17], dtype=np.int64))
    specs = precompute._build_patch_specs(sample=sample, bag_id=bag_id, full_coords=coords, selected_indices=np.array([0]))
    np.testing.assert_array_equal(precompute._encode_selected_patch_specs(patch_specs=specs), [17])

    monkeypatch.setattr(sish_rgb, "_build_slide_processor", lambda **kwargs: processor)
    monkeypatch.setattr(processor, "read_patch_region", lambda wsi, **kwargs: crop(wsi=wsi))
    strategy = sish_rgb.SISHRGB(params={"n_clusters": 1}, config=cfg)
    monkeypatch.setattr(sish_rgb, "load_sish_trash_classifier", lambda **kwargs: SimpleNamespace(predict=lambda rows: np.zeros(len(rows), dtype=int)))
    result = strategy.run(np.ones((1, 4), dtype=np.float32), sample=sample, coords=coords, histogram_rgb=np.ones((1, 768)), slide_lengths=[1], tiling_id=bag_id)
    assert result.data.shape == (1, 4)
    assert len(opened) == len(closed) == 3
    assert all(obj.properties.mpp == 0.25 for obj in opened)
