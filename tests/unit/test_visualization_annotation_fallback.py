from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pathforge.slide_retrieval.visualization.service import (
    SlideRetrievalVisualizationService,
)


def make_service(tmp_path, rows):
    service = SlideRetrievalVisualizationService.__new__(SlideRetrievalVisualizationService)
    service.annotations_df = pd.DataFrame(rows)
    service.dataset_cfg_by_name = {
        name: SimpleNamespace(
            name=name, slides_dir=tmp_path / name, artifacts_dir=tmp_path / name
        )
        for name in ("a", "b")
    }
    return service


def test_visualization_uses_shared_constructor_in_all_source_reads(tmp_path, monkeypatch):
    staged = tmp_path / "renamed.svs"
    staged.write_bytes(b"fake")
    service = make_service(tmp_path, [{
        "dataset": "a", "slide": "S1", "patient": "P1", "category": "tumor",
        "fallback_mpp": "0.25", "wsi_path": str(staged),
    }])
    asset = service._resolve_slide_asset("S1")
    assert asset.slide_path == staged
    assert asset.fallback_mpp == 0.25
    loaded = []
    closed = []

    def load(wsi):
        assert wsi.fallback_mpp == 0.25
        assert wsi.patient == "P1"
        assert wsi.annotations["dataset"] == "a"
        assert wsi.path == staged
        loaded.append(wsi)

    processor = SimpleNamespace(
        load_wsi=load,
        close_wsi=lambda wsi: closed.append(wsi),
        get_base_mpp=lambda wsi: wsi.fallback_mpp,
        get_thumbnail=lambda wsi, **kwargs: (np.zeros((2, 2, 3), dtype=np.uint8), 1.0, 1.0),
        read_patch_region=lambda *args: np.zeros((2, 2, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(service, "_build_processor", lambda: processor)
    assert service._load_base_mpp(asset) == 0.25
    assert service._load_thumbnail_and_spec(asset) is not None
    patches = service._load_patch_strip_images(
        asset=asset,
        selected_coords=np.array([[0, 0]]),
        coords_array=np.array([[0, 0, 2, 2, 0]]),
    )
    assert len(patches) == 1
    assert len(loaded) == len(closed) == 3


def test_visualization_rejects_ambiguous_slide_ids_and_accepts_dataset(tmp_path):
    service = make_service(tmp_path, [
        {"dataset": "a", "slide": "S1", "fallback_mpp": 0.25},
        {"dataset": "b", "slide": "S1", "fallback_mpp": 0.5},
    ])
    with pytest.raises(ValueError, match="Ambiguous"):
        service._resolve_slide_asset("S1")
    assert service._resolve_slide_asset("S1", dataset_name="b").fallback_mpp == 0.5
    assert service._resolve_slide_asset("missing") is None
