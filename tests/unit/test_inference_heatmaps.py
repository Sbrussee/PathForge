from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pathbench.core.io.h5.base import FileHandleH5
from pathbench.core.io.h5 import heatmaps as heatmap_io
from pathbench.core.io.h5 import tiles as tiles_io
from pathbench.inference.heatmaps import create_inference_heatmap
from pathbench.utils.registries import EXPLAINERS


@dataclass(frozen=True)
class _FakeHeatmap:
    coords: torch.Tensor
    scores: torch.Tensor


class _FakeHeatmapExplainer:
    def initialize(self, config):
        self.config = config

    def explain(self, input):
        coords = input["coords"]
        scores = input["instance_scores"].float()
        if "mask" in input:
            mask = input["mask"].bool()
            coords = coords[mask]
            scores = scores[mask]
        scores = (scores - scores.min()) / torch.clamp(scores.max() - scores.min(), min=1e-12)
        return _FakeHeatmap(coords=coords, scores=scores)


def _register_fake_explainer_once(name: str) -> None:
    if EXPLAINERS.is_available(name):
        return
    EXPLAINERS.register(name)(_FakeHeatmapExplainer)


def test_create_inference_heatmap_persists_h5_and_json_sidecar(tmp_path):
    _register_fake_explainer_once("fake_inference_heatmap")
    artifact_path = tmp_path / "slide.h5"
    scores_path = tmp_path / "scores.npy"
    output_path = tmp_path / "heatmap.json"
    np.save(scores_path, np.asarray([0.2, 0.5, 1.0], dtype=np.float32))

    with FileHandleH5(artifact_path, mode="a") as slide_artifact:
        tiles_io.write_coords(
            slide_artifact,
            "256px_0.5mpp",
            np.asarray(
                [
                    [0, 0, 256, 256, 0],
                    [256, 0, 256, 256, 0],
                    [512, 0, 256, 256, 0],
                ],
                dtype=np.int32,
            ),
        )

    result = create_inference_heatmap(
        artifact_path=artifact_path,
        bag_id="256px_0.5mpp",
        scores_path=scores_path,
        heatmap_backend="fake_inference_heatmap",
        heatmap_name="attention",
        output_path=output_path,
        model_path="model.ckpt",
    )

    assert result.num_points == 3
    assert output_path.exists()
    with FileHandleH5(artifact_path, mode="r") as slide_artifact:
        heatmap = heatmap_io.read_prediction_heatmap(slide_artifact, "256px_0.5mpp", "attention")

    assert heatmap["coords"].shape == (3, 2)
    assert heatmap["scores"].shape == (3,)
    assert heatmap["metadata"]["backend"] == "fake_inference_heatmap"


def test_create_inference_heatmap_applies_mask(tmp_path):
    _register_fake_explainer_once("fake_inference_heatmap")
    artifact_path = tmp_path / "slide.h5"
    scores_path = tmp_path / "scores.npy"
    coords_path = tmp_path / "coords.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(scores_path, np.asarray([0.2, 0.5, 1.0], dtype=np.float32))
    np.save(coords_path, np.asarray([[0, 0], [256, 0], [512, 0]], dtype=np.float32))
    np.save(mask_path, np.asarray([1, 0, 1], dtype=np.uint8))

    result = create_inference_heatmap(
        artifact_path=artifact_path,
        bag_id="256px_0.5mpp",
        scores_path=scores_path,
        coords_path=coords_path,
        mask_path=mask_path,
        heatmap_backend="fake_inference_heatmap",
        heatmap_name="masked_attention",
    )

    assert result.num_points == 2
    with FileHandleH5(artifact_path, mode="r") as slide_artifact:
        heatmap = heatmap_io.read_prediction_heatmap(slide_artifact, "256px_0.5mpp", "masked_attention")

    assert heatmap["coords"].tolist() == [[0.0, 0.0], [512.0, 0.0]]
