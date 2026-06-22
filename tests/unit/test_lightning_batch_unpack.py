import torch

from pathbench.training.lightning import LightningTrainer
from pathbench.training.lightning import LightningModuleAdapter
from tests.smoke._smoke_training import make_training_config


class _ToyModel(torch.nn.Module):
    def forward_bag(self, bag, mask=None, coords=None, label=None):
        del mask, coords, label
        return torch.zeros(bag.shape[0], 2)


class _ToyLoss(torch.nn.Module):
    def forward(self, preds, target):
        del preds, target
        return torch.tensor(0.0)


class _Config:
    pass


def test_unpack_batch_accepts_canonical_dict():
    adapter = LightningModuleAdapter(_ToyModel(), _ToyLoss(), _Config())
    batch = {
        "X": torch.zeros(2, 3, 4),
        "Y": torch.tensor([1, 0]),
        "mask": torch.ones(2, 3, dtype=torch.bool),
    }

    bag, target, kwargs = adapter._unpack_batch(batch)

    assert bag.shape == (2, 3, 4)
    assert target.tolist() == [1, 0]
    assert kwargs["mask"].shape == (2, 3)


def test_native_trainer_uses_padded_collate_for_multi_bag_batches(tmp_path):
    cfg = make_training_config(
        tmp_path / "native_batching",
        task="classification",
        epochs=1,
        lr=1e-3,
        dropout=0.0,
        batch_size=2,
    )

    trainer = LightningTrainer(cfg)

    assert trainer._collate_fn() is not None


class _SingleBagDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        del index
        return {
            "X": torch.zeros(2, 4, dtype=torch.float32),
            "Y": torch.tensor(1.25, dtype=torch.float32),
        }


class _ToyTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward_bag(self, bag, mask=None, coords=None, label=None):
        del mask, coords, label
        return self.bias.expand(bag.shape[0], 1)


def test_lightning_trainer_fit_supports_non_classification_without_classification_config(
    monkeypatch, tmp_path
):
    cfg = make_training_config(
        tmp_path / "regression_training",
        task="regression",
        epochs=1,
        lr=1e-3,
        dropout=0.0,
        batch_size=1,
    )
    cfg.classification = None
    trainer = LightningTrainer(cfg)
    trainer.trainer.fit = lambda *args, **kwargs: None
    trainer._save_validation_artifacts = lambda *args, **kwargs: None
    trainer.checkpoint_callback.best_model_path = ""
    trainer.checkpoint_callback.best_model_score = torch.tensor(0.25)

    best_path, best_score = trainer.fit(
        _ToyTrainModel(),
        _SingleBagDataset(),
        _SingleBagDataset(),
        _ToyLoss(),
    )

    assert best_path == ""
    assert best_score == 0.25
