import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np



class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate




class Siren(nn.Module):
    #TODO : CITE SIREN paper. Adapted from their official github repo.
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))
        
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        # self.relu = nn.LeakyReLU(1e-5)
        self.relu = nn.ReLU()
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            self.final_linear_seg = nn.Linear(hidden_features, int(out_features/2))
            self.final_linear_intensities = nn.Linear(hidden_features, int(out_features/2))

            
            with torch.no_grad():
                self.final_linear_seg.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0)
                self.final_linear_intensities.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0)
                pass
        else:
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        
        
        self.net = nn.Sequential(*self.net)
        # print("Input features = ", in_features)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        final_linear_seg_pre = self.final_linear_seg(output)
        final_linear_seg = self.softmax(final_linear_seg_pre) #To ensure sum of probabilities = 1
        final_linear_intensities_pre = self.final_linear_intensities(output)
        final_linear_intensities = self.relu(final_linear_intensities_pre) #Normalizing the output image intensities. Note that this needs normalizing ground truth HF Image as well
        # final_linear_intensities = self.sigmoid(final_linear_intensities_pre)
        
        return final_linear_seg_pre


# ====== Coordinate Grid Generator ======

def generate_coord_grid_3d(depth, height, width, device):
    """Generates normalized [-1,1] coordinate grid for 3D volume"""
    z = torch.linspace(-1, 1, depth, device=device)
    y = torch.linspace(-1, 1, height, device=device)
    x = torch.linspace(-1, 1, width, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    coords = torch.stack([xx, yy, zz], dim=-1)  # [D, H, W, 3]
    coords = coords.view(-1, 3)  # Flatten → [N, 3]
    return coords


# ====== Example Synthetic Data ======

def generate_synthetic_volume(D, H, W):
    """Make a synthetic smooth 3D volume for testing"""
    z = torch.linspace(-math.pi, math.pi, D)
    y = torch.linspace(-math.pi, math.pi, H)
    x = torch.linspace(-math.pi, math.pi, W)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    vol = torch.sin(xx) * torch.cos(yy) * torch.sin(zz)  # [D, H, W]
    return (vol - vol.min()) / (vol.max() - vol.min())  # normalize to [0,1]


# ====== Training Super-Resolution ======

def train_siren_sr():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # High-res ground truth (synthetic example)
    D_hr, H_hr, W_hr = 192, 174 , 174
    volume_hr = generate_synthetic_volume(D_hr, H_hr, W_hr).to(device)

    # Low-res training target (simulate downsampled scan)
    D_lr, H_lr, W_lr = 96, 87, 87
    volume_lr = F.interpolate(volume_hr.unsqueeze(0).unsqueeze(0),
                              size=(D_lr, H_lr, W_lr),
                              mode='trilinear',
                              align_corners=True).squeeze()

    # Prepare training coordinates and intensities
    coords_lr = generate_coord_grid_3d(D_lr, H_lr, W_lr, device).unsqueeze(0)  # [1, N,3]

    target_lr = volume_lr.view(-1, 1)  # [N,1]

    # Model
    model = Siren(in_features=3, hidden_features=256, hidden_layers=3, out_features=8, outermost_linear=True, 
                 first_omega_0=30, hidden_omega_0=30.).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # Training loop
    epochs = 5
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(coords_lr.unsqueeze(0)).squeeze(0)  # [N,1]
        print("pred shape: ", pred.shape, "target shape: ",  target_lr.shape)
        loss = loss_fn(pred[0,:,0], target_lr)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  Loss: {loss.item():.6f}")

    # ===== Super-resolution inference =====
    coords_hr = generate_coord_grid_3d(D_hr, H_hr, W_hr, device).unsqueeze(0)
    with torch.no_grad():
        sr_pred = model(coords_hr)
        sr_pred = sr_pred[0,:,0].view(D_hr, H_hr, W_hr)

    print("Super-resolution output shape:", sr_pred.shape)
    return sr_pred.cpu()


if __name__ == "__main__":
    sr_volume = train_siren_sr()
