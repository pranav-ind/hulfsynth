
import torch, torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
from torchmetrics.functional.image import image_gradients
from Utils.utils import  gradient, laplace, get_device, get_metrics
from Models.layers import FinerLayer, SineLayer



class Finer(nn.Module):
    #TODO : CITE FINER paper. Adapted from their official github repo
    def __init__(self, in_features=2, out_features=3, hidden_layers=3, hidden_features=256, 
                 first_omega=30, hidden_omega=30, 
                 init_method='sine', init_gain=1, fbs=None, hbs=None, 
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.net = []
        self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(1e-5)
        self.softmax = nn.Softmax(dim=2)
        self.net.append(FinerLayer(in_features, hidden_features, is_first=True, 
                                   omega=first_omega, 
                                   init_method=init_method, init_gain=init_gain, fbs=fbs,
                                   alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        for i in range(hidden_layers):
            self.net.append(FinerLayer(hidden_features, hidden_features, 
                                       omega=hidden_omega, 
                                       init_method=init_method, init_gain=init_gain, hbs=hbs,
                                       alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        
        
        self.net.append(FinerLayer(hidden_features, out_features, is_last=True, omega=hidden_omega,  hbs=hbs)) # omega: For weight init 
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        output_seg = output[:,:,0:4] #Extracting segmentation outputs from final layer
        # seg_stacked = torch.stack((output_seg[0,:,0].view(size), output_seg[0,:,1].view(size), output_seg[0,:,2].view(size), output_seg[0,:,3].view(size)), dim=0).unsqueeze(0)
        output_seg_real = self.softmax(output_seg)
        # output_img = self.relu(output[:,:,4:])
        output_img = self.sigmoid(output[:,:,4:])
        return output_seg, output_seg_real, output_img, coords



#Modified SIREN 

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
        
        return final_linear_seg_pre, final_linear_seg, final_linear_intensities_pre, final_linear_intensities, coords