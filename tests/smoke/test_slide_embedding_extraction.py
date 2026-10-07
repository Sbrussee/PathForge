"""Run both embedding levels through the full policy on one sample WSI."""

import numpy as np
import pytest

from ._smoke_dataset import build_gtex_smoke_annotations, capture_smoke_metrics


@pytest.mark.smoke
def test_slide_embedding_policy_and_cache_on_sample_wsi(monkeypatch, tmp_path):
    """Measure real tile I/O, normalization, encoding and H5 caching without weights."""
    import torch
    from huggingface_hub import hf_hub_download
    from lazyslide_models import MODEL_REGISTRY
    from torchvision.transforms.v2 import Compose, ToDtype, ToImage

    from pathforge.config.config import Config
    from pathforge.core.experiments.base import Experiment
    from pathforge.core.experiments.combinations import ComboConfig
    from pathforge.core.experiments.combo_ids import build_feature_name, build_tiling_id
    from pathforge.core.io.slide_artifacts.base import FileHandleH5
    from pathforge.core.io.slide_artifacts import features as features_io, tiles as tiles_io
    from pathforge.core.slide_processing.lazyslide import catalog
    from pathforge.policy.feature_extraction import FeatureExtractionPolicy
    from pathforge.utils.test_samples import download_gtex_slides

    class ImageEncoder:
        task = "vision"
        name = "smoke_image"
        batches = 0

        def __init__(self, model_path=None, token=None):
            pass

        def to(self, device):
            return self

        def get_transform(self):
            return Compose([ToImage(), ToDtype(torch.float32, scale=True)])

        def encode_image(self, images):
            ImageEncoder.batches += 1
            return images.mean(dim=(2, 3))

    class SlideEncoder:
        task = "slide_encoder"
        vision_encoder = "smoke_image"
        calls = 0

        def to(self, device):
            return self

        def encode_slide(self, embeddings, coords=None, **kwargs):
            SlideEncoder.calls += 1
            assert embeddings.shape[1] == coords.shape[1]
            assert kwargs["base_tile_size"] > 0
            return {"embeddings": embeddings.mean(dim=1)}

    monkeypatch.setitem(MODEL_REGISTRY, "smoke_image", ImageEncoder)
    monkeypatch.setitem(MODEL_REGISTRY, "smoke_slide_encoder", SlideEncoder)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    catalog.lazyslide_model_names.cache_clear()
    slide_id = "GTEX-111YS-2226"
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    slide_path = hf_hub_download(
        "RendeiroLab/LazySlide-data", f"gtex_artery_data/{slide_id}.svs", repo_type="dataset",
    )
    (slides_dir / f"{slide_id}.svs").symlink_to(slide_path)
    metadata_path = tmp_path / "metadata.csv"
    download_gtex_slides().to_csv(metadata_path, index=False)
    annotations = build_gtex_smoke_annotations(metadata_path, slide_ids=[slide_id], strict=False)
    annotations["dataset"] = "sample"
    annotations_path = tmp_path / "annotations.csv"
    annotations.to_csv(annotations_path, index=False)
    names = ["smoke_image", "smoke_slide_encoder-slide"]
    try:
        config = Config.model_validate({
            "experiment": {
                "project_name": "slide_embeddings", "mode": "feature_extraction",
                "annotation_file": str(annotations_path), "project_root": str(tmp_path / "project"),
                "report": False, "thumbnail": False,
            },
            "slide_processing": {
                "backend": "lazyslide", "segmentation_method": "otsu",
                "feature_extraction": {"batch_size": 16, "num_workers": 0, "amp": False},
            },
            "datasets": [{
                "name": "sample", "slides_dir": str(slides_dir),
                "artifacts_dir": str(tmp_path / "artifacts"), "used_for": "all",
            }],
            "benchmark_parameters": {
                "tile_px": [256], "tile_mpp": [1.0], "color_norm": ["reinhard"],
                "feature_extraction": names, "mil": [],
            },
        })
        policy = FeatureExtractionPolicy(Experiment(config))
        with capture_smoke_metrics(tmp_path / "metrics", step_name="patch_and_slide_embeddings"):
            assert policy.execute()["status"] == "feature_extraction_done"
        with FileHandleH5(tmp_path / "artifacts" / f"{slide_id}.h5", mode="r") as artifact:
            matrices = []
            for name in names:
                combo = ComboConfig(tile_px=256, tile_mpp=1.0, feature_extraction=name, color_norm="reinhard")
                tiling_id = build_tiling_id(combo)
                matrix = features_io.read_features(artifact, tiling_id, build_feature_name(combo))
                assert matrix.shape[1] == 3
                assert matrix.dtype == np.float32
                assert np.isfinite(matrix).all()
                matrices.append(matrix)
            assert matrices[0].shape[0] == tiles_io.coords_num_rows(artifact, tiling_id)
            assert matrices[1].shape == (1, 3)
            np.testing.assert_allclose(matrices[1], matrices[0].mean(axis=0, keepdims=True))
        batches = ImageEncoder.batches
        assert SlideEncoder.calls == 1
        with capture_smoke_metrics(tmp_path / "metrics", step_name="slide_embedding_cache_reuse"):
            policy.execute()
        assert ImageEncoder.batches == batches
        assert SlideEncoder.calls == 1
    finally:
        catalog.lazyslide_model_names.cache_clear()
