from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pathbench.core.explainer_base import ExplainerBase
from pathbench.utils.optional.torchmil import require_torchmil
from pathbench.utils.registries import EXPLAINERS


@dataclass(frozen=True)
class HeatMap:
    """Prediction heatmap payload produced from instance-level scores.

    Attributes:
        coords: Tensor shaped ``[N, 2]`` with x/y tile coordinates in pixels.
        scores: Tensor shaped ``[N]`` with finite normalized importance values in
            ``[0, 1]``.
    """

    coords: torch.Tensor
    scores: torch.Tensor


class TorchMILHeatmapExplainer(ExplainerBase):
    """Explainer adapter for TorchMIL instance scores.

    Input:
        ``explain`` expects a dictionary with ``coords`` shaped ``[N, 2]`` and
        either ``instance_scores``, ``attention``, or ``scores`` shaped ``[N]``.
        Optional ``mask`` shaped ``[N]`` removes padded instances.

    Output:
        HeatMap with finite scores min-max normalized to ``[0, 1]``. The caller
        can persist this under a prediction heatmap H5 namespace.

    Example:
        ```python
        explainer = TorchMILHeatmapExplainer()
        heatmap = explainer.explain({
            "coords": torch.tensor([[0, 0], [256, 0]]),
            "instance_scores": torch.tensor([0.2, 0.8]),
        })
        assert heatmap.scores.tolist() == [0.0, 1.0]
        ```
    """

    def initialize(self, config: dict[str, Any]) -> None:
        require_torchmil("TorchMIL heatmap explainer")

    def explain(self, input: Any) -> HeatMap:
        require_torchmil("TorchMIL heatmap explainer")
        if not isinstance(input, dict):
            raise TypeError("TorchMIL heatmap explainer expects a dictionary input.")
        coords = input.get("coords")
        scores = input.get("instance_scores", input.get("attention", input.get("scores")))
        if not isinstance(coords, torch.Tensor) or coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("Heatmap coords must be a tensor shaped [N, 2].")
        if not isinstance(scores, torch.Tensor) or scores.ndim != 1:
            raise ValueError("Heatmap scores must be a tensor shaped [N].")
        if "mask" in input and input["mask"] is not None:
            mask = input["mask"].bool()
            coords = coords[mask]
            scores = scores[mask]
        scores = scores.float()
        if not torch.isfinite(scores).all():
            raise ValueError("Heatmap scores contain NaN or Inf.")
        score_min = scores.min()
        score_max = scores.max()
        denom = torch.clamp(score_max - score_min, min=1e-12)
        normalized = (scores - score_min) / denom
        return HeatMap(coords=coords, scores=normalized)


def register_torchmil_heatmap_explainer() -> None:
    """Register the TorchMIL heatmap explainer when TorchMIL is available."""

    if not EXPLAINERS.is_available("torchmil_heatmap"):
        EXPLAINERS.register("torchmil_heatmap")(TorchMILHeatmapExplainer)
