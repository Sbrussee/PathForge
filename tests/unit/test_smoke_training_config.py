from __future__ import annotations

from pathlib import Path

from tests.smoke._smoke_training import (
    DEFAULT_SMOKE_BATCH_SIZE,
    DEFAULT_SMOKE_EPOCHS,
    _smoke_batch_size,
    make_training_config,
)


def test_make_training_config_uses_task_specific_monitor(tmp_path: Path) -> None:
    classification_cfg = make_training_config(
        tmp_path / "classification",
        task="classification",
        epochs=3,
        lr=1e-3,
        dropout=0.0,
    )
    survival_cfg = make_training_config(
        tmp_path / "survival",
        task="survival",
        epochs=3,
        lr=1e-3,
        dropout=0.0,
    )

    assert classification_cfg.experiment.mode == "benchmark"
    assert classification_cfg.mil.best_epoch_based_on == "balanced_accuracy"
    assert survival_cfg.mil.best_epoch_based_on == "c_index"


def test_make_training_config_preserves_requested_epochs_and_batch_size(
    tmp_path: Path,
) -> None:
    cfg = make_training_config(
        tmp_path / "custom",
        task="survival_discrete",
        epochs=5,
        lr=5e-4,
        dropout=0.1,
        batch_size=4,
    )

    assert cfg.mil.epochs == 5
    assert cfg.mil.batch_size == 4
    assert cfg.mil.best_epoch_based_on == "c_index"


def test_smoke_defaults_increase_epochs_and_classification_batch_size() -> None:
    dataset = [object(), object(), object()]

    assert DEFAULT_SMOKE_EPOCHS >= 5
    assert _smoke_batch_size("classification", dataset) == min(
        DEFAULT_SMOKE_BATCH_SIZE,
        len(dataset),
    )
    assert _smoke_batch_size("survival", dataset) == len(dataset)
