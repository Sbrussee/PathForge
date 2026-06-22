from __future__ import annotations

from typing import Any, Literal

import torch


TaskName = Literal["classification", "regression", "survival", "survival_discrete"]


def normalize_torchmil_output(output: Any, *, task: TaskName | str) -> torch.Tensor:
    """Normalize backend model outputs to PathBench loss contracts.

    Args:
        output: Tensor or mapping emitted by a TorchMIL model. Supported mapping
            keys are checked in order: ``logits``, ``Y_hat``, ``pred``,
            ``prediction``, ``risk``, ``hazards``, and ``survival``.
        task: Current PathBench task. Classification expects logits shaped
            ``[B, C]`` or ``[B]``. Continuous survival expects risk/log-hazard
            shaped ``[B]`` or ``[B, 1]``. Discrete survival expects hazards or
            logits shaped ``[B, T]``.

    Returns:
        torch.Tensor: Normalized finite tensor matching PathBench task losses.

    Example:
        ```python
        logits = normalize_torchmil_output({"logits": torch.zeros(2, 3)}, task="classification")
        assert logits.shape == (2, 3)
        ```

    Raises:
        TypeError: If no tensor output can be extracted.
        AssertionError: If shape or finite-value checks fail.
    """

    tensor = _extract_tensor(output)
    assert tensor.is_floating_point(), "Backend output must be floating point logits/risk."
    assert torch.isfinite(tensor).all(), "Backend output contains NaN or Inf."

    if task == "classification":
        assert tensor.ndim in {1, 2}, "Classification output must have shape [B] or [B, C]."
        return tensor
    if task == "survival":
        assert tensor.ndim in {1, 2}, "Continuous survival output must have shape [B] or [B, 1]."
        if tensor.ndim == 2:
            assert tensor.shape[1] == 1, "Continuous survival output [B, C] requires C=1."
            return tensor.reshape(-1)
        return tensor
    if task == "survival_discrete":
        assert tensor.ndim == 2, "Discrete survival output must have shape [B, T]."
        assert tensor.shape[1] > 0, "Discrete survival output requires at least one time bin."
        return tensor
    if task == "regression":
        assert tensor.ndim in {1, 2}, "Regression output must have shape [B] or [B, K]."
        if tensor.ndim == 2:
            assert tensor.shape[1] == 1, "Regression output [B, K] requires K=1 for continuous regression."
            return tensor.reshape(-1)
        return tensor
    raise ValueError(f"Unsupported task for TorchMIL output normalization: {task!r}")


def _extract_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("logits", "Y_hat", "pred", "prediction", "risk", "hazards", "survival"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError("TorchMIL backend output did not contain a tensor prediction.")
