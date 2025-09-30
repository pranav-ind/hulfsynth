import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------
# 1. Define a simple LightningModule
# ----------------------------
class LitClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28 * 28, 128)
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.layer1(x))
        return self.layer2(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# ----------------------------
# 2. Dummy dataset & dataloader
# ----------------------------
x = torch.randn(1000, 1, 28, 28)
y = torch.randint(0, 10, (1000,))
train_loader = DataLoader(TensorDataset(x, y), batch_size=32)


# ----------------------------
# 3. Trainer with multiple GPUs
# ----------------------------
# Example: use 2 GPUs on a single node
trainer = pl.Trainer(
    accelerator="gpu",  # tells Lightning to use CUDA
    devices=2,          # number of GPUs to use
    strategy="ddp",     # DistributedDataParallel (best practice)
    max_epochs=5
)

model = LitClassifier()
trainer.fit(model, train_loader)
