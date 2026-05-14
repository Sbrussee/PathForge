from __future__ import annotations

import torch

from pathbench.adapters.torchmil.backend import TorchMILBackendModel
from pathbench.training.metrics import save_task_evaluation_artifacts


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
        "pathbench.adapters.torchmil.backend.require_torchmil",
        lambda feature: None,
    )
    monkeypatch.setattr(
        "pathbench.adapters.torchmil.backend.build_torchmil_model",
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
