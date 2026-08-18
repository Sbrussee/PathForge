from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Iterable, Protocol

import torch
import torch.nn as nn

from pathforge.adapters.torchmil.task_output import normalize_torchmil_output
from pathforge.core.datasets.bag_schema import BagBatch, assert_bag_schema
from pathforge.core.models.mil_base import MILModelBase
from pathforge.utils.optional.torchmil import load_torchmil_modules, require_torchmil
from pathforge.utils.registries import MODELS


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
        task_types: PathForge tasks supported by this model specification.
        required_keys: Batch keys routed to the backend model. ``X`` is always
            required by the canonical schema.
        build_kwargs: Default constructor kwargs merged before config kwargs.

    Example:
        .. code-block:: python

            spec = TorchMILModelSpec(name="ABMIL", task_types=("classification",))
            model = build_torchmil_model(spec, {"in_shape": (1024,), "out_shape": 2})

    """

    name: str
    task_types: tuple[str, ...] = ("classification", "regression", "survival")
    required_keys: tuple[str, ...] = ("X",)
    build_kwargs: dict[str, Any] = field(default_factory=dict)


TORCHMIL_MODEL_SPECS: dict[str, TorchMILModelSpec] = {
    "ABMIL": TorchMILModelSpec(name="ABMIL"),
    "DSMIL": TorchMILModelSpec(name="DSMIL"),
    "CLAM": TorchMILModelSpec(name="CLAM"),
    "TransMIL": TorchMILModelSpec(name="TransMIL"),
    "PatchGCN": TorchMILModelSpec(name="PatchGCN", required_keys=("X", "adj")),
}


def resolve_torchmil_model_spec(model_name: str) -> TorchMILModelSpec:
    """Resolve a TorchMIL model specification by name.

    Unknown names are allowed as generic TorchMIL model class names so long as
    the installed ``torchmil.models`` module exposes them. This preserves the
    single-adapter design without requiring one PathForge class per model.

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
    signature = inspect.signature(model_factory)
    if not any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        kwargs = {
            name: value
            for name, value in kwargs.items()
            if name in signature.parameters
        }
    model = model_factory(**kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError(f"TorchMIL model '{spec.name}' did not return a torch.nn.Module.")
    return model


class TorchMILBackendModel(MILModelBase):
    """Generic PathForge adapter for TorchMIL MIL models.

    Args:
        torchmil_model: Name of the TorchMIL model class in ``torchmil.models``.
        task: PathForge task name. ``classification`` expects logits, ``survival``
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
        .. code-block:: python

            model = TorchMILBackendModel(
                torchmil_model="ABMIL",
                task="classification",
                torchmil_model_kwargs={"in_shape": (1024,), "out_shape": 2},
            )
            logits = model.forward_bag(torch.zeros(2, 8, 1024), mask=torch.ones(2, 8, dtype=torch.bool))


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
        graph_enabled: bool = False,
        graph_neighbor_space: str = "spatial",
        graph_k: int = 8,
        graph_symmetric: bool = True,
        graph_self_loops: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        require_torchmil("MIL backend 'torchmil'")
        self.task = task
        self.graph_enabled = graph_enabled
        self.graph_neighbor_space = graph_neighbor_space
        self.graph_k = graph_k
        self.graph_symmetric = graph_symmetric
        self.graph_self_loops = graph_self_loops
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
        elif self.graph_enabled or "adj" in self.spec.required_keys:
            if self.graph_neighbor_space == "spatial":
                if coords is None:
                    raise ValueError(
                        f"Graph model '{self.spec.name}' uses spatial neighbors "
                        "and therefore requires tile coordinates."
                    )
                graph_vectors = coords.float()
            elif self.graph_neighbor_space == "feature":
                graph_vectors = bag.float()
            else:
                raise ValueError(
                    f"Unsupported graph neighbor space: {self.graph_neighbor_space!r}."
                )
            batch["adj"] = _knn_adjacency(
                graph_vectors,
                mask=mask,
                k=self.graph_k,
                symmetric=self.graph_symmetric,
                self_loops=self.graph_self_loops,
            )
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
        # Route only model inputs (never the label ``Y``). Modern TorchMIL
        # models (>=1.0) take the bag features ``X`` as a positional tensor with
        # optional ``mask``/``coords``/``adj`` keyword arguments.
        routed = {key: batch[key] for key in ("X", "mask", "coords", "adj") if key in batch}
        bag = routed["X"]
        extra = {key: value for key, value in routed.items() if key != "X"}
        forward_signature = inspect.signature(self.backend_model.forward)
        if not any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in forward_signature.parameters.values()
        ):
            extra = {
                name: value
                for name, value in extra.items()
                if name in forward_signature.parameters
            }
        try:
            return self.backend_model(bag, **extra)
        except TypeError:
            # Fallbacks for backends that consume the full mapping or only ``X``.
            try:
                return self.backend_model(routed)
            except (TypeError, AttributeError):
                return self.backend_model(bag)

    def get_learnable_parameters(self) -> Iterable[torch.nn.Parameter]:
        return (param for param in self.parameters() if param.requires_grad)


def _knn_adjacency(
    vectors: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    k: int = 8,
    symmetric: bool = True,
    self_loops: bool = True,
) -> torch.Tensor:
    """Build a deterministic symmetric k-NN graph from vector distances.

    ``vectors`` may contain spatial coordinates or feature embeddings. Edges
    connect the nearest vectors by Euclidean distance and include self-loops.
    """

    if vectors.ndim != 3:
        raise ValueError(f"Expected vectors shape [B, N, D], got {tuple(vectors.shape)}.")
    distances = torch.cdist(vectors.float(), vectors.float())
    batch_size, instances, _ = distances.shape
    valid = (
        mask.bool()
        if mask is not None
        else torch.ones((batch_size, instances), dtype=torch.bool, device=vectors.device)
    )
    pair_valid = valid.unsqueeze(1) & valid.unsqueeze(2)
    distances = distances.masked_fill(~pair_valid, float("inf"))
    eye = torch.eye(instances, dtype=torch.bool, device=vectors.device).unsqueeze(0)
    distances = distances.masked_fill(eye, float("inf"))
    neighbour_count = min(k, max(instances - 1, 1))
    indices = distances.topk(neighbour_count, dim=-1, largest=False).indices
    adjacency = torch.zeros_like(distances, dtype=vectors.dtype)
    adjacency.scatter_(2, indices, 1.0)
    if symmetric:
        adjacency = torch.maximum(adjacency, adjacency.transpose(1, 2))
    adjacency = adjacency * pair_valid.to(adjacency.dtype)
    if self_loops:
        adjacency = adjacency + torch.diag_embed(valid.to(adjacency.dtype))
    return adjacency


def register_torchmil_backend() -> None:
    """Register the generic backend and concrete TorchMIL model names."""

    if not MODELS.is_available("torchmil"):
        MODELS.register("torchmil")(TorchMILBackendModel)
    for model_name in TORCHMIL_MODEL_SPECS:
        if not MODELS.is_available(model_name):
            MODELS.register(model_name)(TorchMILBackendModel)
