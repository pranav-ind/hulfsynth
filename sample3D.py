import torch
import torch.nn as nn
import torch.nn.functional as F


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


# Example usage
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example input: batch of 1, 1 channel, 64x64x64 volume
    x = torch.randn((1, 1, 224, 224, 160)).to(device)
    y = torch.randn((1, 1, 224, 224, 160)).to(device)  # high-quality target

    model = UNet3D().to(device)
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    # Loss function
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, y)

    print("Loss:", loss.item())

    
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model.train()

for epoch in range(10):  # replace with dataloader for real training
    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
