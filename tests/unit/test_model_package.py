from __future__ import annotations

import json

import torch

from pathbench.cli.inference import main as inference_main
from pathbench.core.io.h5.base import FileHandleH5
from pathbench.core.io.h5.features import write_features
from pathbench.inference.model_package import (
    load_packaged_model,
    predict_bag,
    save_packaged_model,
)
from pathbench.utils.registries import populate_dynamic_registries
from tests.smoke._smoke_training import make_training_config


def test_packaged_model_roundtrip_preserves_predictions(tmp_path) -> None:
    populate_dynamic_registries()
    config = make_training_config(
        tmp_path / "roundtrip",
        task="classification",
        epochs=1,
        lr=1e-3,
        dropout=0.0,
    )
    torch.manual_seed(7)
    from pathbench.utils.registries import MODELS

    model = MODELS.get("VarMIL")(input_dim=4, hidden_dim=8, output_dim=2)
    bag = torch.randn(1, 6, 4)
    package_path = save_packaged_model(
        path=tmp_path / "model_package.pt",
        model=model,
        config=config,
        model_name="VarMIL",
        input_dim=4,
        output_dim=2,
    )

    loaded = load_packaged_model(package_path)
    expected = predict_bag(model.eval(), bag, task="classification")
    actual = predict_bag(loaded.model, bag, task=loaded.task)

    torch.testing.assert_close(actual, expected)
    assert loaded.model_name == "VarMIL"
    assert loaded.inference_metadata["bag_id"] == "224px_1.0mpp"
    assert loaded.inference_metadata["feature_extractor"] == "resnet18"


def test_inference_cli_reads_packaged_model_and_h5(tmp_path) -> None:
    populate_dynamic_registries()
    config = make_training_config(
        tmp_path / "cli",
        task="classification",
        epochs=1,
        lr=1e-3,
        dropout=0.0,
    )
    torch.manual_seed(11)
    from pathbench.utils.registries import MODELS

    model = MODELS.get("VarMIL")(input_dim=4, hidden_dim=8, output_dim=2)
    package_path = save_packaged_model(
        path=tmp_path / "cli_model_package.pt",
        model=model,
        config=config,
        model_name="VarMIL",
        input_dim=4,
        output_dim=2,
    )
    h5_path = tmp_path / "slide.h5"
    output_path = tmp_path / "prediction.json"
    features = torch.randn(5, 4).numpy()
    with FileHandleH5(h5_path, mode="a") as handle:
        write_features(
            handle,
            bag_id="224px_1.0mpp",
            extractor_name="resnet18",
            feature_matrix=features,
        )

    exit_code = inference_main(
        [
            "--model_path",
            str(package_path),
            "--input",
            str(h5_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["task"] == "classification"
    assert payload["model_name"] == "VarMIL"
    assert payload["bag_id"] == "224px_1.0mpp"
    assert payload["feature_extractor"] == "resnet18"
    assert len(payload["predictions"]) == 2
    assert len(payload["probs"]) == 2
    assert payload["predicted_class"] in {0, 1}
