from __future__ import annotations

import pytest
import torch

from pathbench.adapters.mil_lab.backend import (
    MILLabBackendModel,
    _canonicalize_mil_lab_kwargs,
    normalize_mil_lab_output,
)
from pathbench.config.config import Config
from pathbench.training.metrics import save_task_evaluation_artifacts
from pathbench.utils import registries as registries_module
from tests.conftest import DUMMY_FE, DUMMY_MIL


class _FakeMILLabModel(torch.nn.Module):
    def forward(
        self,
        bag: torch.Tensor,
        *,
        loss_fn=None,
        label=None,
        return_attention: bool = False,
        return_slide_feats: bool = False,
    ):
        _ = (return_attention, return_slide_feats)
        logits = bag.mean(dim=1)
        results = {"logits": logits}
        if loss_fn is not None and label is not None:
            results["loss"] = loss_fn(logits, label)
        return results, {"logits": logits.detach().cpu().numpy()}


def test_canonicalize_mil_lab_kwargs_maps_pathbench_names() -> None:
    kwargs = _canonicalize_mil_lab_kwargs(
        {"input_dim": 1024, "output_dim": 3, "hidden_dim": 256}
    )

    assert kwargs["in_dim"] == 1024
    assert kwargs["num_classes"] == 3
    assert kwargs["embed_dim"] == 256
    assert "input_dim" not in kwargs
    assert "output_dim" not in kwargs
    assert "hidden_dim" not in kwargs


def test_normalize_mil_lab_output_handles_tuple_results() -> None:
    logits = torch.randn(2, 3)
    output = ({"logits": logits}, {"ignored": True})

    normalized = normalize_mil_lab_output(output)

    assert isinstance(normalized, dict)
    assert torch.equal(normalized["logits"], logits)


def test_mil_lab_backend_model_wraps_results_dict(monkeypatch) -> None:
    monkeypatch.setattr(
        "pathbench.adapters.mil_lab.backend.require_mil_lab",
        lambda feature: None,
    )
    monkeypatch.setattr(
        "pathbench.adapters.mil_lab.backend.build_mil_lab_model",
        lambda spec, config_kwargs, from_pretrained=False: _FakeMILLabModel(),
    )

    model = MILLabBackendModel(
        mil_lab_model="abmil",
        mil_lab_model_kwargs={"input_dim": 4, "output_dim": 4},
    )
    bag = torch.ones((2, 5, 4), dtype=torch.float32)
    label = torch.tensor([1, 0])
    loss_fn = torch.nn.CrossEntropyLoss()

    output = model.forward_bag(bag, label=label, loss_fn=loss_fn)

    assert isinstance(output, dict)
    assert output["logits"].shape == (2, 4)
    assert output["loss"].ndim == 0


def test_mil_lab_backend_outputs_support_visualization_artifacts(
    monkeypatch,
    tmp_path,
) -> None:
    class _BinaryFakeMILLabModel(torch.nn.Module):
        def forward(
            self,
            bag: torch.Tensor,
            *,
            loss_fn=None,
            label=None,
            return_attention: bool = False,
            return_slide_feats: bool = False,
        ):
            _ = (loss_fn, label, return_attention, return_slide_feats)
            pooled = bag.mean(dim=1)
            logits = torch.stack([pooled[:, 0], pooled[:, 1]], dim=1)
            return {"logits": logits}, {"ignored": True}

    monkeypatch.setattr(
        "pathbench.adapters.mil_lab.backend.require_mil_lab",
        lambda feature: None,
    )
    monkeypatch.setattr(
        "pathbench.adapters.mil_lab.backend.build_mil_lab_model",
        lambda spec, config_kwargs, from_pretrained=False: _BinaryFakeMILLabModel(),
    )

    model = MILLabBackendModel(
        mil_lab_model="abmil",
        mil_lab_model_kwargs={"input_dim": 4, "output_dim": 2},
    )
    bag = torch.tensor(
        [
            [[3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0]],
            [[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([0, 1], dtype=torch.long)

    output = model.forward_bag(bag, label=target, loss_fn=torch.nn.CrossEntropyLoss())
    assert isinstance(output, dict)

    artifacts = save_task_evaluation_artifacts(
        output["logits"],
        target,
        task="classification",
        output_dir=tmp_path,
        prefix="val",
    )

    assert artifacts.figure_paths["confusion_matrix"].exists()
    assert artifacts.figure_paths["roc_auc_curve"].exists()
    assert artifacts.figure_paths["pr_auc_curve"].exists()
    assert artifacts.figure_paths["calibration_curve"].exists()


def test_config_requires_mil_lab_model_when_backend_selected(
    minimal_benchmark_config: dict[str, object],
    monkeypatch,
) -> None:
    cfg_dict = dict(minimal_benchmark_config)
    cfg_dict["mil"] = {"backend": "mil-lab"}
    monkeypatch.setattr("pathbench.config.config.is_mil_lab_available", lambda: True)

    with pytest.raises(ValueError, match="mil\\.mil_lab_model"):
        Config.model_validate(cfg_dict)


def test_config_accepts_mil_lab_backend_when_model_is_set(
    minimal_fe_config: dict[str, object],
    monkeypatch,
) -> None:
    cfg_dict = dict(minimal_fe_config)
    cfg_dict["experiment"] = dict(cfg_dict["experiment"], task="classification", mode="benchmark")
    cfg_dict["mil"] = {
        "backend": "mil-lab",
        "mil_lab_model": "abmil.base.uni.none",
        "mil_lab_model_kwargs": {"num_classes": 2},
    }
    cfg_dict["metrics"] = {"classification_backend": "native"}
    cfg_dict["benchmark_parameters"] = {
        "tile_px": [256],
        "tile_mpp": [0.5],
        "feature_extraction": [DUMMY_FE],
        "mil": [DUMMY_MIL],
        "loss": ["CrossEntropyLoss"],
    }
    monkeypatch.setattr("pathbench.config.config.is_mil_lab_available", lambda: True)

    cfg = Config.model_validate(cfg_dict)

    assert cfg.mil.backend == "mil-lab"
    assert cfg.mil.mil_lab_model == "abmil.base.uni.none"


def test_populate_dynamic_registries_imports_remaining_native_models(monkeypatch) -> None:
    monkeypatch.setattr(registries_module, "_populated", False)
    monkeypatch.setattr(registries_module, "is_mil_lab_available", lambda: False)
    monkeypatch.setattr(registries_module, "is_torchmil_available", lambda: False)
    monkeypatch.setattr(registries_module, "is_torchmetrics_available", lambda: False)
    monkeypatch.setattr(registries_module, "is_torchsurv_available", lambda: False)

    registries_module.populate_dynamic_registries()

    assert registries_module.MODELS.is_available("PerceiverMIL")
    assert registries_module.MODELS.is_available("PrototypeMIL")
    assert registries_module.MODELS.is_available("VarMIL")
