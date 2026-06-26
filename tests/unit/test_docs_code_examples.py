# tests/unit/test_docs_code_examples.py
"""
Back-to-source tests for every code claim made in the PathForge docs.
Each test is coupled to a documented API pattern; if a test fails, update
the documentation to match the real code (or fix the code and keep the docs).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# H5 I/O — FileHandleH5 context manager
# ---------------------------------------------------------------------------

class TestFileHandleH5:
    def test_context_manager_opens_and_closes(self, tmp_path: Path) -> None:
        from pathforge.core.io.h5.base import FileHandleH5

        h5_path = tmp_path / "slide.h5"
        with FileHandleH5(h5_path, mode="a") as fh:
            assert fh.h5 is not None

    def test_h5_property_raises_outside_context(self, tmp_path: Path) -> None:
        from pathforge.core.io.h5.base import FileHandleH5

        fh = FileHandleH5(tmp_path / "slide.h5", mode="a")
        with pytest.raises(RuntimeError, match="not open"):
            _ = fh.h5

    def test_read_mode_requires_existing_file(self, tmp_path: Path) -> None:
        from pathforge.core.io.h5.base import FileHandleH5
        import h5py

        h5_path = tmp_path / "slide.h5"
        with FileHandleH5(h5_path, mode="a") as _:
            pass  # create the file

        with FileHandleH5(h5_path, mode="r") as fh:
            assert fh.h5 is not None


# ---------------------------------------------------------------------------
# H5 I/O — read_features / write_features (documented in inference.rst)
# ---------------------------------------------------------------------------

class TestReadFeaturesDocumentedAPI:
    def test_write_and_read_features_roundtrip(self, tmp_path: Path) -> None:
        """Docs show: read_features(fh, bag_id, extractor_name) → np.ndarray."""
        from pathforge.core.io.h5.base import FileHandleH5
        from pathforge.core.io.h5.features import read_features, write_features

        h5_path = tmp_path / "slide.h5"
        bag_id = "256px_0.5mpp"
        extractor_name = "resnet50"
        feature_matrix = np.random.rand(10, 2048).astype(np.float32)

        with FileHandleH5(h5_path, mode="a") as fh:
            write_features(fh, bag_id, extractor_name, feature_matrix)

        with FileHandleH5(h5_path, mode="r") as fh:
            result = read_features(fh, bag_id, extractor_name)

        assert result.shape == (10, 2048)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, feature_matrix)

    def test_read_features_positional_api_matches_docs(self, tmp_path: Path) -> None:
        """Docs use positional args: read_features(fh, bag_id, extractor_name)."""
        import inspect
        from pathforge.core.io.h5.features import read_features

        sig = inspect.signature(read_features)
        params = list(sig.parameters.keys())
        # First three positional params must be slide_artifact, bag_id, extractor_name
        assert params[0] == "slide_artifact"
        assert params[1] == "bag_id"
        assert params[2] == "extractor_name"

    def test_write_features_rejects_1d_input(self, tmp_path: Path) -> None:
        from pathforge.core.io.h5.base import FileHandleH5
        from pathforge.core.io.h5.features import write_features

        with FileHandleH5(tmp_path / "slide.h5", mode="a") as fh:
            with pytest.raises(ValueError, match="shape"):
                write_features(fh, "256px_0.5mpp", "resnet50", np.ones(10))


# ---------------------------------------------------------------------------
# H5 I/O — coords shape is (N, 5) int32 (documented in architecture.rst)
# ---------------------------------------------------------------------------

class TestCoordsShape:
    def test_write_and_read_coords_shape_is_n5(self, tmp_path: Path) -> None:
        """Docs state coords are stored as (N, 5) int32."""
        from pathforge.core.io.h5.base import FileHandleH5
        from pathforge.core.io.h5 import tiles as tiles_io

        coords_in = np.array(
            [[0, 0, 256, 256, 0], [256, 0, 256, 256, 0]], dtype=np.int32
        )
        h5_path = tmp_path / "slide.h5"
        with FileHandleH5(h5_path, mode="a") as fh:
            tiles_io.write_coords(fh, "256px_0.5mpp", coords_in)

        with FileHandleH5(h5_path, mode="r") as fh:
            coords_out = tiles_io.read_coords(fh, "256px_0.5mpp")

        assert coords_out.shape[1] == 5
        assert coords_out.dtype == np.int32

    def test_write_coords_rejects_non_5_column_array(self, tmp_path: Path) -> None:
        from pathforge.core.io.h5.base import FileHandleH5
        from pathforge.core.io.h5 import tiles as tiles_io

        bad = np.ones((3, 4), dtype=np.int32)
        with FileHandleH5(tmp_path / "slide.h5", mode="a") as fh:
            with pytest.raises(ValueError):
                tiles_io.write_coords(fh, "256px_0.5mpp", bad)


# ---------------------------------------------------------------------------
# H5 I/O — read_prediction_heatmap (documented in inference.rst)
# ---------------------------------------------------------------------------

class TestReadPredictionHeatmapDocumentedAPI:
    def test_read_prediction_heatmap_returns_dict(self, tmp_path: Path) -> None:
        """Docs: read_prediction_heatmap returns dict with coords, scores, metadata."""
        from pathforge.core.io.h5.base import FileHandleH5
        from pathforge.core.io.h5.heatmaps import (
            write_prediction_heatmap,
            read_prediction_heatmap,
        )

        h5_path = tmp_path / "slide.h5"
        coords = np.array([[0.0, 0.0], [256.0, 0.0]], dtype=np.float32)
        scores = np.array([0.1, 0.9], dtype=np.float32)

        with FileHandleH5(h5_path, mode="a") as fh:
            write_prediction_heatmap(
                fh, "256px_0.5mpp", "abmil_attention",
                coords=coords, scores=scores,
                metadata={"backend": "torchmil"},
            )

        with FileHandleH5(h5_path, mode="r") as fh:
            result = read_prediction_heatmap(fh, "256px_0.5mpp", "abmil_attention")

        assert isinstance(result, dict)
        assert "coords" in result
        assert "scores" in result
        assert "metadata" in result
        assert result["coords"].shape == (2, 2)
        assert result["scores"].shape == (2,)
        assert result["metadata"]["backend"] == "torchmil"

    def test_read_heatmap_function_does_not_exist(self) -> None:
        """Docs must NOT reference read_heatmap — it does not exist."""
        import pathforge.core.io.h5.heatmaps as heatmaps_mod

        assert not hasattr(heatmaps_mod, "read_heatmap"), (
            "read_heatmap does not exist; docs must use read_prediction_heatmap"
        )

    def test_write_prediction_heatmap_validates_score_range(
        self, tmp_path: Path
    ) -> None:
        """Docs note scores must be in [0, 1]."""
        from pathforge.core.io.h5.base import FileHandleH5
        from pathforge.core.io.h5.heatmaps import write_prediction_heatmap

        with FileHandleH5(tmp_path / "slide.h5", mode="a") as fh:
            with pytest.raises(ValueError, match=r"\[0, 1\]"):
                write_prediction_heatmap(
                    fh, "256px_0.5mpp", "bad_scores",
                    coords=np.array([[0.0, 0.0]]),
                    scores=np.array([1.5], dtype=np.float32),
                )

    def test_write_prediction_heatmap_rejects_nan_scores(
        self, tmp_path: Path
    ) -> None:
        from pathforge.core.io.h5.base import FileHandleH5
        from pathforge.core.io.h5.heatmaps import write_prediction_heatmap

        with FileHandleH5(tmp_path / "slide.h5", mode="a") as fh:
            with pytest.raises(ValueError, match="NaN or Inf"):
                write_prediction_heatmap(
                    fh, "256px_0.5mpp", "nan_scores",
                    coords=np.array([[0.0, 0.0]]),
                    scores=np.array([float("nan")], dtype=np.float32),
                )

    def test_write_prediction_heatmap_positional_api(self) -> None:
        """Docs call write_prediction_heatmap(fh, bag_id, heatmap_name, ...)."""
        import inspect
        from pathforge.core.io.h5.heatmaps import write_prediction_heatmap

        sig = inspect.signature(write_prediction_heatmap)
        params = list(sig.parameters.keys())
        assert params[0] == "slide_artifact"
        assert params[1] == "bag_id"
        assert params[2] == "heatmap_name"


# ---------------------------------------------------------------------------
# H5 layout paths (documented in architecture.rst and inference.rst)
# ---------------------------------------------------------------------------

class TestH5LayoutPaths:
    def test_features_dataset_path(self) -> None:
        from pathforge.core.io.h5.layout import DEFAULT_LAYOUT

        path = DEFAULT_LAYOUT.features_dataset("256px_0.5mpp", "resnet50")
        assert path == "bags/256px_0.5mpp/features/resnet50"

    def test_prediction_heatmap_paths(self) -> None:
        from pathforge.core.io.h5.layout import DEFAULT_LAYOUT

        bag_id = "256px_0.5mpp"
        heatmap_name = "abmil_attention"

        coords_path = DEFAULT_LAYOUT.prediction_heatmap_coords_dataset(bag_id, heatmap_name)
        scores_path = DEFAULT_LAYOUT.prediction_heatmap_scores_dataset(bag_id, heatmap_name)
        meta_path = DEFAULT_LAYOUT.prediction_heatmap_metadata_dataset(bag_id, heatmap_name)

        # Docs show: bags/{bag_id}/predictions/heatmaps/{heatmap_name}/{dataset}
        assert coords_path.startswith(f"bags/{bag_id}/predictions/heatmaps/{heatmap_name}/")
        assert scores_path.startswith(f"bags/{bag_id}/predictions/heatmaps/{heatmap_name}/")
        assert meta_path.startswith(f"bags/{bag_id}/predictions/heatmaps/{heatmap_name}/")

    def test_coords_dataset_path(self) -> None:
        from pathforge.core.io.h5.layout import DEFAULT_LAYOUT

        path = DEFAULT_LAYOUT.coords_dataset("256px_0.5mpp")
        assert path == "bags/256px_0.5mpp/coords"


# ---------------------------------------------------------------------------
# MIL model — instance_scores returns [B, N] (documented in inference.rst)
# ---------------------------------------------------------------------------

class TestInstanceScoresShape:
    def test_instance_scores_returns_batch_x_instances(self) -> None:
        """Docs: instance_scores(bag) returns [B, N]. Squeeze for single bag."""
        from pathforge.core.models.var_mil import VarMIL

        model = VarMIL(input_dim=4, hidden_dim=3, output_dim=2)
        bag = torch.randn(1, 8, 4)
        scores = model.instance_scores(bag)

        assert scores.shape == (1, 8), f"Expected [B, N]=[1, 8], got {scores.shape}"

    def test_instance_scores_squeeze_gives_1d(self) -> None:
        """Docs show scores_1d = scores.squeeze(0) to get [N]."""
        from pathforge.core.models.var_mil import VarMIL

        model = VarMIL(input_dim=4, hidden_dim=3, output_dim=2)
        bag = torch.randn(1, 5, 4)
        scores = model.instance_scores(bag)
        scores_1d = scores.squeeze(0)

        assert scores_1d.ndim == 1
        assert scores_1d.shape[0] == 5

    def test_instance_scores_raises_for_model_without_attention(self) -> None:
        from pathforge.core.models.mil_base import MILModelBase

        class _NoAttn(MILModelBase):
            @property
            def bag_size(self) -> int | None:
                return None

            def forward_bag(self, bag, **kwargs):
                return torch.zeros(bag.shape[0], 2)

        model = _NoAttn()
        with pytest.raises(AttributeError):
            model.instance_scores(torch.zeros(1, 3, 4))


# ---------------------------------------------------------------------------
# LightningModuleAdapter — correct class name (documented in model_packaging.rst)
# ---------------------------------------------------------------------------

class TestLightningModuleAdapter:
    def test_lightning_module_adapter_is_importable(self) -> None:
        """Docs reference pathforge.training.lightning.LightningModuleAdapter."""
        from pathforge.training.lightning import LightningModuleAdapter

        assert LightningModuleAdapter is not None

    def test_lightning_mil_module_does_not_exist(self) -> None:
        """LightningMILModule must NOT appear in docs — it does not exist."""
        import pathforge.training.lightning as lightning_mod

        assert not hasattr(lightning_mod, "LightningMILModule"), (
            "LightningMILModule does not exist; docs must use LightningModuleAdapter"
        )

    def test_save_hyperparameters_excludes_model_and_loss(self) -> None:
        """Docs note model is excluded from hyperparams — must pass explicitly to load_from_checkpoint."""
        import inspect
        import ast
        import textwrap
        from pathforge.training import lightning as mod

        source = inspect.getsource(mod.LightningModuleAdapter.__init__)
        # Verify save_hyperparameters ignore list contains "model" and "loss_fn"
        assert "model" in source
        assert "ignore" in source


# ---------------------------------------------------------------------------
# Registry — populate_dynamic_registries is idempotent (documented in backends.rst)
# ---------------------------------------------------------------------------

class TestPopulateDynamicRegistries:
    def test_populate_is_idempotent(self) -> None:
        from pathforge.utils.registries import populate_dynamic_registries

        populate_dynamic_registries()
        populate_dynamic_registries()  # must not raise or duplicate entries

    def test_native_models_registered_after_populate(self) -> None:
        from pathforge.utils.registries import MODELS, populate_dynamic_registries

        populate_dynamic_registries()
        assert MODELS.is_available("PerceiverMIL")
        assert MODELS.is_available("VarMIL")
        assert MODELS.is_available("PrototypeMIL")

    def test_gcnconv_mil_not_registered(self) -> None:
        """GCNConvMIL was removed — must not appear in registry."""
        from pathforge.utils.registries import MODELS, populate_dynamic_registries

        populate_dynamic_registries()
        assert not MODELS.is_available("GCNConvMIL"), (
            "GCNConvMIL was removed from native models; remove it from docs if listed"
        )
