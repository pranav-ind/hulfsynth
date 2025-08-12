import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchio as tio
from torchio.data import UniformSampler, Queue
import numpy as np

# -------------------------
# SIREN implementation
# -------------------------
class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, w0=1.0, c=6.0, is_first=False):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights(c)

    def init_weights(self, c):
        with torch.no_grad():
            if self.is_first:
                # first layer: uniform(-1/in, 1/in)
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = math.sqrt(c / self.in_features) / self.w0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


class SIREN(nn.Module):
    def __init__(self, in_features=3, out_features=1, hidden_features=256, hidden_layers=3, w0=30.0, w0_hidden=1.0):
        super().__init__()
        layers = [SIRENLayer(in_features, hidden_features, w0=w0, is_first=True)]
        for _ in range(hidden_layers):
            layers.append(SIRENLayer(hidden_features, hidden_features, w0=w0_hidden))
        self.net = nn.Sequential(*layers)
        self.final_linear = nn.Linear(hidden_features, out_features)
        # final weight init (small)
        self.final_linear.weight.uniform_(-math.sqrt(6 / hidden_features) / w0_hidden,
                                          math.sqrt(6 / hidden_features) / w0_hidden)

    def forward(self, coords):
        # coords: (B, N, 3)
        B, N, C = coords.shape
        x = coords.view(B * N, C)
        x = self.net(x)
        x = self.final_linear(x)  # (B*N, out)
        x = x.view(B, N, -1)      # (B, N, out)
        return x                  # (B, N, out)

# -------------------------
# Helper: create normalized grid for a patch
# -------------------------
def make_patch_coords(pd, ph, pw, device):
    # local coords in [-1, 1] over patch grid
    z = torch.linspace(-1.0, 1.0, pd, device=device)
    y = torch.linspace(-1.0, 1.0, ph, device=device)
    x = torch.linspace(-1.0, 1.0, pw, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')  # shape (pd, ph, pw)
    coords = torch.stack([xx, yy, zz], dim=-1)           # (pd, ph, pw, 3)
    coords = coords.view(-1, 3)                          # (N, 3)
    return coords  # CPU or device tensor

# -------------------------
# Example: dataset using TorchIO
# -------------------------
def make_subjects_from_numpy(low_vols, high_vols):
    """
    low_vols, high_vols: lists of numpy arrays with shape (D,H,W) or (C,D,H,W)
    Returns a list of tio.Subject for SubjectsDataset
    """
    subjects = []
    for i, (low, high) in enumerate(zip(low_vols, high_vols)):
        # ensure shape (1, D, H, W)
        if low.ndim == 3:
            low_t = np.expand_dims(low, 0)
        else:
            low_t = low
        if high.ndim == 3:
            high_t = np.expand_dims(high, 0)
        else:
            high_t = high

        subject = tio.Subject(
            low = tio.ScalarImage(tensor=torch.tensor(low_t, dtype=torch.float32)),
            high = tio.ScalarImage(tensor=torch.tensor(high_t, dtype=torch.float32)),
        )
        subjects.append(subject)
    return subjects

# -------------------------
# Training function (patchwise using TorchIO queue)
# -------------------------
def train_siren_with_torchio(low_vols, high_vols,
                             patch_size=(32,32,32),
                             patches_per_volume=20,
                             queue_max_length=128,
                             batch_size=4,
                             device='cuda' if torch.cuda.is_available() else 'cpu',
                             epochs=200,
                             lr=1e-4):
    # 1) Build subjects dataset
    subjects = make_subjects_from_numpy(low_vols, high_vols)
    subjects_dataset = tio.SubjectsDataset(subjects)

    # 2) Create a Patch sampler and Queue (random patches)
    sampler = UniformSampler(patch_size)
    queue = Queue(
        subjects_dataset,
        max_length=queue_max_length,        # max number of items the queue stores
        samples_per_volume=patches_per_volume,
        sampler=sampler,
        num_workers=0,                      # change as needed
    )

    loader = DataLoader(queue, batch_size=batch_size)

    # 3) Model
    model = SIREN(in_features=3, out_features=1, hidden_features=128, hidden_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # precompute patch-local coords on device
    pd, ph, pw = patch_size
    coords_patch = make_patch_coords(pd, ph, pw, device=device)  # (N, 3)
    N = coords_patch.shape[0]

    print("Training starting; patches per vol:", patches_per_volume, "patch_size:", patch_size)

    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0
        for batch in loader:
            # batch is a dict; images accessible as batch['low'][tio.DATA] and batch['high'][tio.DATA]
            # shapes: [B, C, pd, ph, pw]
            low_patches = batch['low'][tio.DATA].to(device)   # (B,1,pd,ph,pw)
            high_patches = batch['high'][tio.DATA].to(device) # (B,1,pd,ph,pw)

            B = low_patches.shape[0]
            # Flatten target: (B, N, 1)
            targets = high_patches.view(B, 1, -1).permute(0,2,1)  # (B,N,1)

            # Make coords batch: (B,N,3)
            coords_batched = coords_patch.unsqueeze(0).expand(B, -1, -1)  # (B,N,3)

            optimizer.zero_grad()
            preds = model(coords_batched)  # (B,N,1)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        epoch_loss = running_loss / max(1, n_batches)
        if (epoch+1) % max(1, epochs//10) == 0 or epoch==0:
            print(f"Epoch {epoch+1}/{epochs}  loss: {epoch_loss:.6f}")

    print("Training finished.")
    return model

# -------------------------
# Super-resolution inference (query HR coords)
# -------------------------
def infer_highres_from_model(model, D_hr, H_hr, W_hr, device='cuda' if torch.cuda.is_available() else 'cpu', bs_coords=200000):
    # Generate grid coords for HR volume (normalized in [-1,1] global)
    z = torch.linspace(-1.0, 1.0, D_hr, device=device)
    y = torch.linspace(-1.0, 1.0, H_hr, device=device)
    x = torch.linspace(-1.0, 1.0, W_hr, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')  # (D,H,W)
    coords = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)  # (N,3)
    N = coords.shape[0]

    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        # chunk coords to avoid memory blowup
        for i in range(0, N, bs_coords):
            coords_chunk = coords[i:i+bs_coords].unsqueeze(0).to(device)  # (1, M, 3)
            out_chunk = model(coords_chunk)  # (1, M, 1)
            preds.append(out_chunk.squeeze(0).cpu())
    preds = torch.cat(preds, dim=0)  # (N, 1)
    vol = preds.view(D_hr, H_hr, W_hr)
    return vol

# -------------------------
# Example run with synthetic data
# -------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Make a synthetic high-res volume and then downsample to low-res
    D_hr, H_hr, W_hr = 64, 64, 64
    vol_hr = np.zeros((D_hr, H_hr, W_hr), dtype=np.float32)
    # Example pattern
    coords = np.indices((D_hr, H_hr, W_hr)).astype(np.float32)
    x = coords[2] / W_hr * 2 * math.pi
    y = coords[1] / H_hr * 2 * math.pi
    z = coords[0] / D_hr * 2 * math.pi
    vol_hr = (np.sin(x) * np.cos(y) * np.sin(z)).astype(np.float32)
    vol_hr = (vol_hr - vol_hr.min()) / (vol_hr.max() - vol_hr.min())

    # low-res by trilinear downsample
    import torch.nn.functional as F
    vol_hr_t = torch.tensor(vol_hr).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    vol_lr_t = F.interpolate(vol_hr_t, size=(32,32,32), mode='trilinear', align_corners=True)
    vol_lr = vol_lr_t.squeeze().numpy()

    # For demonstration we create lists of paired volumes (here just duplicates)
    low_vols = [vol_lr, vol_lr]   # two volumes to make a small dataset
    high_vols = [vol_hr, vol_hr]

    # Train (this will be relatively slow; shorten epochs for test)
    model = train_siren_with_torchio(low_vols, high_vols,
                                     patch_size=(32,32,32),
                                     patches_per_volume=10,
                                     queue_max_length=64,
                                     batch_size=4,
                                     device=device,
                                     epochs=100,
                                     lr=1e-4)

    # Inference: generate HR estimate (querying a HR grid)
    sr = infer_highres_from_model(model, D_hr, H_hr, W_hr, device=device)
    print("SR shape:", sr.shape)
