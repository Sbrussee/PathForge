import torch
from pathbench.core.models.mil_base import MILModel


class B(MILModel):
pass


def test_mil_forward_contract():
model = B(embed_dim=4)
bag = [torch.zeros(4) for _ in range(8)]
out = model(bag)
assert out.shape[-1] == 2