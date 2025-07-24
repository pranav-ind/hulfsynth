import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from load_data3D import get_dataset
import wandb


wandb.login()
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super(UNet3D, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.decoder = nn.ModuleList()
        self.upconv = nn.ModuleList()

        # Encoder path
        for feature in features:
            self.encoder.append(DoubleConv3D(in_channels, feature))
            in_channels = feature

        # Decoder path
        for feature in reversed(features):
            self.upconv.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv3D(feature * 2, feature))

        # Bottleneck
        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)

        # Final output layer
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder)):
            x = self.upconv[idx](x)
            skip_connection = skip_connections[idx]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx](x)

        return self.final_conv(x)

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample_x = self.x[idx]
        sample_y = self.y[idx]

        return sample_x, sample_y



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    dataset = get_dataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    model = UNet3D().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 5
    loss_fn = nn.MSELoss()
    model = UNet3D().to(device)

    # print(next(iter(dataloader))[0].shape)

    for epoch in trange(num_epochs):  # replace with dataloader for real training
        running_loss = 0.0
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            print("batch_x_y shape: ", batch_x.shape, batch_y.shape)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        wandb.log({"total_loss": running_loss}, mode='L') 



        
