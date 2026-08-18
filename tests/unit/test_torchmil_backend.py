from __future__ import annotations

import torch

from pathforge.adapters.torchmil.backend import (
    TorchMILBackendModel,
    TorchMILModelSpec,
    _knn_adjacency,
    build_torchmil_model,
)
from pathforge.training.metrics import save_task_evaluation_artifacts


class _FakeTorchMILModel(torch.nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        bag = batch["X"].float()
        pooled = bag.mean(dim=1)
        logits = torch.stack([pooled[:, 0], pooled[:, 1]], dim=1)
        return {"logits": logits}


def test_torchmil_backend_outputs_support_visualization_artifacts(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "pathforge.adapters.torchmil.backend.require_torchmil",
        lambda feature: None,
    )
    monkeypatch.setattr(
        "pathforge.adapters.torchmil.backend.build_torchmil_model",
        lambda spec, config_kwargs: _FakeTorchMILModel(),
    )

    model = TorchMILBackendModel(
        torchmil_model="ABMIL",
        task="classification",
        torchmil_model_kwargs={"in_shape": (4,), "out_shape": 2},
    )
    bag = torch.tensor(
        [
            [[4.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]],
            [[0.0, 5.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]],
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


def test_build_torchmil_model_filters_unsupported_constructor_kwargs(monkeypatch) -> None:
    class _Factory(torch.nn.Module):
        def __init__(self, in_shape: tuple[int, ...]) -> None:
            super().__init__()
            self.in_shape = in_shape

    modules = type("Modules", (), {"models": type("Models", (), {"Factory": _Factory})})
    monkeypatch.setattr(
        "pathforge.adapters.torchmil.backend.load_torchmil_modules",
        lambda: modules,
    )
    model = build_torchmil_model(
        TorchMILModelSpec(name="Factory"),
        {"in_shape": (16,), "out_shape": 2},
    )
    assert model.in_shape == (16,)


def test_feature_knn_adjacency_is_symmetric_and_masks_padding() -> None:
    bag = torch.tensor([[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.0, 0.0]]])
    mask = torch.tensor([[True, True, True, False]])
    adjacency = _knn_adjacency(bag, mask=mask, k=1)
    assert adjacency.shape == (1, 4, 4)
    assert torch.equal(adjacency, adjacency.transpose(1, 2))
    assert adjacency[0, 3].sum() == 0
    assert torch.equal(adjacency[0].diag(), mask[0].float())


def test_patchgcn_builds_missing_adjacency(monkeypatch) -> None:
    class _Patch(torch.nn.Module):
        def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            assert adj.shape == (1, 3, 3)
            return x.mean(dim=(1, 2))

    monkeypatch.setattr(
        "pathforge.adapters.torchmil.backend.require_torchmil",
        lambda feature: None,
    )
    monkeypatch.setattr(
        "pathforge.adapters.torchmil.backend.build_torchmil_model",
        lambda spec, config_kwargs: _Patch(),
    )
    model = TorchMILBackendModel(
        torchmil_model="PatchGCN",
        task="survival",
        torchmil_model_kwargs={},
    )
    output = model.forward_bag(
        torch.randn(1, 3, 4),
        coords=torch.tensor([[[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]]]),
    )
    assert output.shape == (1,)


def test_graph_enabled_builds_adjacency_for_generic_graph_model(monkeypatch) -> None:
    """User graph settings provide ``adj`` for models outside the static catalog."""

    class _GraphModel(torch.nn.Module):
        def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            assert adj.shape == (1, 3, 3)
            assert torch.equal(adj, adj.transpose(1, 2))
            assert torch.equal(adj[0].diag(), torch.ones(3))
            return x.mean(dim=(1, 2))

    monkeypatch.setattr(
        "pathforge.adapters.torchmil.backend.require_torchmil",
        lambda feature: None,
    )
    monkeypatch.setattr(
        "pathforge.adapters.torchmil.backend.build_torchmil_model",
        lambda spec, config_kwargs: _GraphModel(),
    )
    monkeypatch.setattr(
        "pathforge.adapters.torchmil.backend.resolve_torchmil_model_spec",
        lambda name: TorchMILModelSpec(name=name),
    )
    model = TorchMILBackendModel(
        torchmil_model="CustomGraphMIL",
        task="survival",
        graph_enabled=True,
        graph_neighbor_space="feature",
        graph_k=1,
        graph_symmetric=True,
        graph_self_loops=True,
    )

    output = model.forward_bag(torch.randn(1, 3, 4))

    assert output.shape == (1,)


def test_knn_adjacency_respects_direction_and_self_loop_settings() -> None:
    vectors = torch.tensor([[[0.0], [1.0], [3.0]]])

    adjacency = _knn_adjacency(
        vectors,
        k=1,
        symmetric=False,
        self_loops=False,
    )

    assert torch.equal(adjacency[0].diag(), torch.zeros(3))
    assert not torch.equal(adjacency, adjacency.transpose(1, 2))
