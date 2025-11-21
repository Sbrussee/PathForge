from __future__ import annotations

from pathbench.core.models.model_base import ModelBase

class SlideLevelMILModel(ModelBase):
    """For pre-aggregated slide vectors."""
    def __init__(self, in_dim: int = 256, lr: float = 1e-3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 2))
        self.lr = lr
    def forward(self, x: torch.Tensor):
        return self.net(x)
    def training_step(self, batch, batch_idx):
        logits = self(batch.x)
        loss = torch.nn.functional.cross_entropy(logits, batch.y)
        self.log("train_loss", loss)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)