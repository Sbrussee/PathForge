from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
import torch.nn as nn

from pathbench.core.models.mil_base import MILModelBase
from pathbench.utils.optional.mil_lab import load_mil_lab_modules, require_mil_lab
from pathbench.utils.optional.mil_lab import is_mil_lab_available
from pathbench.utils.optional.torchmil import is_torchmil_available
from pathbench.utils.registries import MODELS


@dataclass(frozen=True)
class MILLabModelSpec:
    """Factory contract for one MIL-Lab model name."""

    name: str
    task_types: tuple[str, ...] = ("classification",)
    build_kwargs: dict[str, Any] = field(default_factory=dict)


MILLAB_MODEL_SPECS: dict[str, MILLabModelSpec] = {
    "abmil": MILLabModelSpec(name="abmil"),
    "clam": MILLabModelSpec(name="clam"),
    "dftd": MILLabModelSpec(name="dftd"),
    "dsmil": MILLabModelSpec(name="dsmil"),
    "ilra": MILLabModelSpec(name="ilra"),
    "rrt": MILLabModelSpec(name="rrt"),
    "transformer": MILLabModelSpec(name="transformer"),
    "transmil": MILLabModelSpec(name="transmil"),
    "wikg": MILLabModelSpec(name="wikg"),
}

MILLAB_NATIVE_ALIASES: dict[str, str] = {
    "AttentionMIL": "abmil",
    "CLAM_SB": "clam",
    "DTFD_MIL": "dftd",
    "DSMIL": "dsmil",
    "ILRA_MIL": "ilra",
    "RRT_MIL": "rrt",
    "TransformerMIL": "transformer",
    "Transformer": "transformer",
    "TransMIL": "transmil",
    "WiKG_MIL": "wikg",
}

TORCHMIL_FALLBACK_ALIASES: dict[str, str] = {
    "AttentionMIL": "ABMIL",
    "CLAM_SB": "CLAM",
    "DTFD_MIL": "DFTDMIL",
    "DSMIL": "DSMIL",
    "TransMIL": "TransMIL",
}


def _canonicalize_mil_lab_kwargs(config_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Map PathBench-style constructor kwargs onto MIL-Lab builder kwargs."""

    kwargs = dict(config_kwargs)
    if "input_dim" in kwargs and "in_dim" not in kwargs:
        kwargs["in_dim"] = kwargs.pop("input_dim")
    if "output_dim" in kwargs and "num_classes" not in kwargs:
        kwargs["num_classes"] = kwargs.pop("output_dim")
    if "hidden_dim" in kwargs and "embed_dim" not in kwargs:
        kwargs["embed_dim"] = kwargs.pop("hidden_dim")
    return kwargs


def build_mil_lab_model(
    spec: MILLabModelSpec,
    config_kwargs: dict[str, Any],
    *,
    from_pretrained: bool = False,
) -> nn.Module:
    """Instantiate a MIL-Lab model from one generic factory path."""

    modules = load_mil_lab_modules()
    create_model = getattr(modules.builder, "create_model", None)
    if create_model is None:
        raise RuntimeError("MIL-Lab builder module does not expose create_model().")

    kwargs = {**spec.build_kwargs, **_canonicalize_mil_lab_kwargs(config_kwargs)}
    model = create_model(spec.name, from_pretrained=from_pretrained, **kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError(f"MIL-Lab model '{spec.name}' did not return a torch.nn.Module.")
    return model


def normalize_mil_lab_output(output: Any) -> torch.Tensor | dict[str, Any]:
    """Normalize MIL-Lab forward output to PathBench expectations."""

    if isinstance(output, tuple):
        results_dict = output[0]
    else:
        results_dict = output

    if isinstance(results_dict, dict):
        logits = results_dict.get("logits")
        if logits is None:
            raise ValueError("MIL-Lab output dict did not contain 'logits'.")
        normalized = {"logits": logits}
        if "loss" in results_dict and results_dict["loss"] is not None:
            normalized["loss"] = results_dict["loss"]
        if "attention" in results_dict:
            normalized["attention"] = results_dict["attention"]
        if "slide_feats" in results_dict:
            normalized["slide_feats"] = results_dict["slide_feats"]
        return normalized

    if isinstance(results_dict, torch.Tensor):
        return results_dict

    raise TypeError("MIL-Lab output was neither a Tensor nor a results dict.")


class MILLabBackendModel(MILModelBase):
    """Generic PathBench adapter for MIL-Lab MIL models."""

    def __init__(
        self,
        *,
        mil_lab_model: str,
        task: str = "classification",
        mil_lab_model_kwargs: dict[str, Any] | None = None,
        mil_lab_from_pretrained: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        require_mil_lab("MIL backend 'mil-lab'")
        self.task = task
        spec = MILLAB_MODEL_SPECS.get(mil_lab_model, MILLabModelSpec(name=mil_lab_model))
        if task not in spec.task_types:
            raise ValueError(
                f"MIL-Lab model '{spec.name}' does not declare support for task '{task}'."
            )
        self.backend_model = build_mil_lab_model(
            spec,
            mil_lab_model_kwargs or {},
            from_pretrained=mil_lab_from_pretrained,
        )

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
        _ = (mask, coords, adj)
        output = self.backend_model(
            bag,
            loss_fn=loss_fn,
            label=label,
            return_attention=False,
            return_slide_feats=False,
        )
        normalized = normalize_mil_lab_output(output)
        if isinstance(normalized, dict):
            if "loss" in normalized or loss_fn is None or label is None:
                return normalized
            logits = normalized["logits"]
            return {"logits": logits, "loss": loss_fn(logits, label)}

        if loss_fn is not None and label is not None:
            return {"logits": normalized, "loss": loss_fn(normalized, label)}
        return normalized

    def get_learnable_parameters(self) -> Iterable[torch.nn.Parameter]:
        return (param for param in self.parameters() if param.requires_grad)


def _make_mil_lab_alias_factory(
    *,
    registry_name: str,
    mil_lab_model: str,
) -> Any:
    def _factory(**kwargs: Any) -> MILLabBackendModel:
        task = kwargs.pop("task", "classification")
        from_pretrained = bool(kwargs.pop("mil_lab_from_pretrained", False))
        return MILLabBackendModel(
            mil_lab_model=mil_lab_model,
            task=task,
            mil_lab_model_kwargs=kwargs,
            mil_lab_from_pretrained=from_pretrained,
        )

    _factory.__name__ = f"{registry_name}MILLabFactory"
    return _factory


def _make_torchmil_alias_factory(
    *,
    registry_name: str,
    torchmil_model: str,
) -> Any:
    from pathbench.adapters.torchmil.backend import TorchMILBackendModel

    def _factory(**kwargs: Any) -> TorchMILBackendModel:
        task = kwargs.pop("task", "classification")
        return TorchMILBackendModel(
            torchmil_model=torchmil_model,
            task=task,
            torchmil_model_kwargs=kwargs,
        )

    _factory.__name__ = f"{registry_name}TorchMILFactory"
    return _factory


def register_mil_lab_backend() -> None:
    """Register the generic MIL-Lab backend and alias names."""

    if not MODELS.is_available("mil-lab"):
        MODELS.register("mil-lab")(MILLabBackendModel)

    for registry_name, model_name in MILLAB_NATIVE_ALIASES.items():
        if MODELS.is_available(registry_name):
            continue
        MODELS.register(registry_name)(
            _make_mil_lab_alias_factory(
                registry_name=registry_name,
                mil_lab_model=model_name,
            )
        )


def register_torchmil_fallback_aliases() -> None:
    """Register TorchMIL aliases for overlapping names when MIL-Lab is unavailable."""

    if is_mil_lab_available():
        return
    if not is_torchmil_available():
        return

    for registry_name, model_name in TORCHMIL_FALLBACK_ALIASES.items():
        if MODELS.is_available(registry_name):
            continue
        MODELS.register(registry_name)(
            _make_torchmil_alias_factory(
                registry_name=registry_name,
                torchmil_model=model_name,
            )
        )
