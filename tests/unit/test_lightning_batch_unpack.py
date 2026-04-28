import torch

from pathbench.training.lightning import LightningModuleAdapter


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


def test_unpack_batch_accepts_legacy_tuple():
    adapter = LightningModuleAdapter(_ToyModel(), _ToyLoss(), _Config())

    bag, target, kwargs = adapter._unpack_batch((torch.zeros(2, 3, 4), torch.tensor([1, 0])))

    assert bag.shape == (2, 3, 4)
    assert target.tolist() == [1, 0]
    assert kwargs == {}


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
