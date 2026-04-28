from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

import torch
import torch.nn as nn

from pathbench.adapters.torchmil.task_output import normalize_torchmil_output
from pathbench.core.datasets.bag_schema import BagBatch, assert_bag_schema
from pathbench.core.models.mil_base import MILModelBase
from pathbench.utils.optional.torchmil import load_torchmil_modules, require_torchmil
from pathbench.utils.registries import MODELS


class MILBackendModelProtocol(Protocol):
    """Protocol implemented by backend MIL model adapters.

    Args:
        bag: Feature tensor shaped ``[B, N, D]`` with finite floating values.
        mask: Optional boolean tensor shaped ``[B, N]`` where true means real
            instance.
        coords: Optional coordinate tensor shaped ``[B, N, 2]``.

    Returns:
        Tensor or mapping containing task predictions. Classification outputs
        are logits shaped ``[B, C]``; survival outputs are risk ``[B]`` or
        discrete hazards ``[B, T]``.
    """

    def forward_bag(
        self,
        bag: torch.Tensor,
        mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        loss_fn: nn.Module | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        ...


@dataclass(frozen=True)
class TorchMILModelSpec:
    """Factory contract for one TorchMIL model name.

    Attributes:
        name: Class or factory name in ``torchmil.models``.
        task_types: PathBench tasks supported by this model specification.
        required_keys: Batch keys routed to the backend model. ``X`` is always
            required by the canonical schema.
        build_kwargs: Default constructor kwargs merged before config kwargs.

    Example:
        ```python
        spec = TorchMILModelSpec(name="ABMIL", task_types=("classification",))
        model = build_torchmil_model(spec, {"in_shape": (1024,), "out_shape": 2})
        ```
    """

    name: str
    task_types: tuple[str, ...] = ("classification",)
    required_keys: tuple[str, ...] = ("X",)
    build_kwargs: dict[str, Any] = field(default_factory=dict)


TORCHMIL_MODEL_SPECS: dict[str, TorchMILModelSpec] = {
    "ABMIL": TorchMILModelSpec(name="ABMIL"),
    "DSMIL": TorchMILModelSpec(name="DSMIL"),
    "CLAM": TorchMILModelSpec(name="CLAM"),
}


def resolve_torchmil_model_spec(model_name: str) -> TorchMILModelSpec:
    """Resolve a TorchMIL model specification by name.

    Unknown names are allowed as generic TorchMIL model class names so long as
    the installed ``torchmil.models`` module exposes them. This preserves the
    single-adapter design without requiring one PathBench class per model.

    Raises:
        RuntimeError: If TorchMIL is not installed.
        ValueError: If the model name is empty.
    """

    if not model_name:
        raise ValueError("mil.torchmil_model must be set when mil.backend='torchmil'.")
    if model_name in TORCHMIL_MODEL_SPECS:
        return TORCHMIL_MODEL_SPECS[model_name]
    require_torchmil("MIL backend 'torchmil'")
    return TorchMILModelSpec(name=model_name)


def build_torchmil_model(spec: TorchMILModelSpec, config_kwargs: dict[str, Any]) -> nn.Module:
    """Instantiate a TorchMIL model from one generic factory path.

    Args:
        spec: Resolved model specification.
        config_kwargs: User-provided constructor kwargs from config.

    Returns:
        nn.Module: Instantiated TorchMIL model.

    Raises:
        ValueError: If ``spec.name`` is not present in ``torchmil.models``.
        RuntimeError: If TorchMIL cannot be imported.
    """

    modules = load_torchmil_modules()
    model_factory = getattr(modules.models, spec.name, None)
    if model_factory is None:
        raise ValueError(f"TorchMIL model '{spec.name}' was not found in torchmil.models.")
    kwargs = {**spec.build_kwargs, **config_kwargs}
    model = model_factory(**kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError(f"TorchMIL model '{spec.name}' did not return a torch.nn.Module.")
    return model


class TorchMILBackendModel(MILModelBase):
    """Generic PathBench adapter for TorchMIL MIL models.

    Args:
        torchmil_model: Name of the TorchMIL model class in ``torchmil.models``.
        task: PathBench task name. ``classification`` expects logits, ``survival``
            expects one risk score per bag, and ``survival_discrete`` expects
            per-bin outputs.
        torchmil_model_kwargs: Constructor kwargs forwarded to the TorchMIL
            model factory. Common examples include ``in_shape=(1024,)`` and
            task-specific output dimensions.

    Input:
        ``forward_bag`` consumes ``bag`` shaped ``[B, N, D]`` with dtype
        ``float32`` or another floating dtype. Optional ``mask`` is shaped
        ``[B, N]`` and optional ``coords`` is shaped ``[B, N, 2]``.

    Output:
        Tensor normalized by task: classification logits ``[B, C]``,
        continuous survival risk ``[B]``, or discrete survival logits/hazards
        ``[B, T]``.

    Example:
        ```python
        model = TorchMILBackendModel(
            torchmil_model="ABMIL",
            task="classification",
            torchmil_model_kwargs={"in_shape": (1024,), "out_shape": 2},
        )
        logits = model.forward_bag(torch.zeros(2, 8, 1024), mask=torch.ones(2, 8, dtype=torch.bool))
        ```

    Raises:
        RuntimeError: If TorchMIL is not installed.
        ValueError: If the selected model/task combination is unsupported.
    """

    def __init__(
        self,
        *,
        torchmil_model: str,
        task: str = "classification",
        torchmil_model_kwargs: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        require_torchmil("MIL backend 'torchmil'")
        self.task = task
        self.spec = resolve_torchmil_model_spec(torchmil_model)
        if task not in self.spec.task_types:
            raise ValueError(
                f"TorchMIL model '{self.spec.name}' does not declare support for task '{task}'."
            )
        self.backend_model = build_torchmil_model(self.spec, torchmil_model_kwargs or {})

    @property
    def bag_size(self) -> int | None:
        return None

    def forward_bag(
        self,
        bag: torch.Tensor,
        mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        loss_fn: nn.Module | None = None,
        adj: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        batch: dict[str, Any] = {"X": bag, "Y": label if label is not None else torch.zeros(bag.shape[0])}
        if mask is not None:
            batch["mask"] = mask.bool()
        if coords is not None:
            batch["coords"] = coords
        if adj is not None:
            batch["adj"] = adj
        assert_bag_schema(batch, batched=True)

        for key in self.spec.required_keys:
            if key not in batch or batch[key] is None:
                raise ValueError(f"TorchMIL model '{self.spec.name}' requires batch key '{key}'.")

        output = self._call_backend(batch)
        normalized = normalize_torchmil_output(output, task=self.task)
        if loss_fn is not None and label is not None:
            return {"logits": normalized, "loss": loss_fn(normalized, label)}
        return normalized

    def _call_backend(self, batch: BagBatch | dict[str, Any]) -> Any:
        routed = {key: batch[key] for key in ("X", "mask", "coords", "adj", "Y") if key in batch}
        try:
            return self.backend_model(routed)
        except TypeError:
            kwargs = {key: value for key, value in routed.items() if key != "X"}
            try:
                return self.backend_model(routed["X"], **kwargs)
            except TypeError:
                return self.backend_model(routed["X"])

    def get_learnable_parameters(self) -> Iterable[torch.nn.Parameter]:
        return (param for param in self.parameters() if param.requires_grad)


def register_torchmil_backend() -> None:
    """Register the generic TorchMIL backend model when TorchMIL is available."""

    if not MODELS.is_available("torchmil"):
        MODELS.register("torchmil")(TorchMILBackendModel)
