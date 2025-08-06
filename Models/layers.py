import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
import time
from torch.nn import init
import math
from tqdm import trange



tonp = lambda x: x.cpu().detach().numpy()
mse_fn = lambda pred, gt: ((pred - gt)**2).mean()
def psnr_fn(pred, gt):
    mse = mse_fn(pred.clip(0, 1), gt)
    if isinstance(pred, torch.Tensor):
        return -10 * torch.log10(mse)
    return -10 * np.log10(mse)

# setup_seed(3407)




def init_weights(m, omega=1, c=1, is_first=False): # Default: Pytorch initialization
    if hasattr(m, 'weight'):
        fan_in = m.weight.size(-1)
        if is_first:
            bound = 1 / fan_in # SIREN
        else:
            bound = math.sqrt(c / fan_in) / omega
        init.uniform_(m.weight, -bound, bound)
        
    
def init_weights_kaiming(m):
    if hasattr(m, 'weight'):
        init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

def init_bias(m, k):
    if hasattr(m, 'bias'):
        # init.uniform_(m.bias, -20, 20)
        init.uniform_(m.bias, -k, k)

'''Used for SIREN, FINER, Gauss, Wire, etc.'''
def init_weights_cond(init_method, linear, omega=1, c=1, is_first=False):
    init_method = init_method.lower()
    if init_method == 'sine':
        init_weights(linear, omega, 6, is_first)    # SIREN initialization
    ## Default: Pytorch initialization

def init_bias_cond(linear, fbs=None, is_first=True):
    if is_first and fbs != None:
        init_bias(linear, fbs)
    ## Default: Pytorch initialization

''' 
    FINER activation
    TODO: alphaType, alphaReqGrad
'''
def generate_alpha(x, alphaType=None, alphaReqGrad=False):
    """
    if alphaType == ...:
        return ...
    """
    with torch.no_grad():
        return torch.abs(x) + 1
    
def finer_activation(x, omega=1, alphaType=None, alphaReqGrad=False):
    return torch.sin(omega * generate_alpha(x, alphaType, alphaReqGrad) * x)


'''
    Gauss. & GF(FINER++Gauss.) activation
'''
def gauss_activation(x, scale):
    return torch.exp(-(scale*x)**2)

def gauss_finer_activation(x, scale, omega, alphaType=None, alphaReqGrad=False):
    return gauss_activation(finer_activation(x, omega, alphaType, alphaReqGrad), scale)

def gauss_finer_activation_norm(x, scale, omega, alphaType=None, alphaReqGrad=False):
    y = gauss_finer_activation(x, scale, omega, alphaType, alphaReqGrad)
    y_min = gauss_activation(torch.tensor(1), scale)
    y_max = gauss_activation(torch.tensor(0), scale)
    return (y - y_min) / (y_max - y_min)

    
'''
    Wire & WF activation
'''
def wire_activation(x, scale, omega_w):
    return torch.exp(1j*omega_w*x - torch.abs(scale*x)**2)

def finer_activation_complex_sep_real_imag(x, omega=1):
    with torch.no_grad():
        alpha_real = torch.abs(x.real) + 1
        alpha_imag = torch.abs(x.imag) + 1
    x.real = x.real * alpha_real
    x.imag = x.imag * alpha_imag
    return torch.sin(omega * x)

def wire_finer_activation(x, scale, omega_w, omega, alphaType=None, alphaReqGrad=False):
    if x.is_complex():
        return wire_activation(finer_activation_complex_sep_real_imag(x, omega), scale, omega_w)
    else:
        return wire_activation(finer_activation(x, omega), scale, omega_w)


## WIRE
class ComplexGaborLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, scale=10, omega_w=20, 
                 is_first=False, is_last=False, 
                 init_method='Pytorch', init_gain=1):
        super().__init__()
        self.scale = scale
        self.omega_w = omega_w
        self.is_last = is_last ## no activation
        dtype = torch.float if is_first else torch.cfloat
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

        # init weights
        init_weights_cond(init_method, self.linear, omega_w, init_gain, is_first)
        
    def forward(self, input):
        wx_b = self.linear(input)
        if not self.is_last:
            return wire_activation(wx_b, self.scale, self.omega_w)
        return wx_b # is_last==True




class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega=30, 
                 is_first=False, is_last=False, 
                 init_method='sine', init_gain=1, fbs=None, hbs=None, 
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.omega = omega
        self.is_last = is_last ## no activation
        self.alphaType = alphaType
        self.alphaReqGrad = alphaReqGrad
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # init weights
        init_weights_cond(init_method, self.linear, omega, init_gain, is_first)
        # init bias
        init_bias_cond(self.linear, fbs, is_first)
    
    def forward(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            return finer_activation(wx_b, self.omega)
        return wx_b # is_last==True

    def forward_with_interm(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            alpha = generate_alpha(wx_b, self.alphaType, self.alphaReqGrad)
            return self.omega*wx_b, self.omega*alpha*wx_b, torch.sin(self.omega*alpha*wx_b)    
        return wx_b # is_last==True
    
          

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






'''

class Finer(nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_layers=3, hidden_features=256, 
                 first_omega=30, hidden_omega=30, 
                 init_method='sine', init_gain=1, fbs=None, hbs=None, 
                 alphaType=None, alphaReqGrad=False):
        super().__init__()
        self.net = []
        self.net.append(FinerLayer(in_features, hidden_features, is_first=True, 
                                   omega=first_omega, 
                                   init_method=init_method, init_gain=init_gain, fbs=fbs,
                                   alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        for i in range(hidden_layers):
            self.net.append(FinerLayer(hidden_features, hidden_features, 
                                       omega=hidden_omega, 
                                       init_method=init_method, init_gain=init_gain, hbs=hbs,
                                       alphaType=alphaType, alphaReqGrad=alphaReqGrad))

        self.net.append(FinerLayer(hidden_features, out_features, is_last=True, 
                                   omega=hidden_omega, 
                                   init_method=init_method, init_gain=init_gain, hbs=hbs)) # omega: For weight init
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        return self.net(coords)
    
    def forward_with_interm(self, input):
        interm = {}
        N = len(self.net)
        for idx, layer in enumerate(self.net):
            if idx != N-1:
                wxb, wxb_finer, sin_activated = layer.forward_with_interm(input)
                interm[f'layer_{idx}_wxb'] = wxb
                interm[f'layer_{idx}_wxb_finer'] = wxb_finer
                interm[f'layer_{idx}_sin_acted'] = sin_activated
                interm[f'layer_{idx}_out'] = sin_activated
                out = sin_activated
            else:
                out = layer(input)
                interm[f'layer_{idx}_out'] = out
            input = out
        return interm

'''


def train_image(model, coords, gt, loss_fn=mse_fn, lr=5e-4, num_epochs=2000, steps_til_summary=10, invnorm=lambda x:x):
    
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / num_epochs, 1))
        
    train_iter = []
    train_psnr = []
    total_time = 0
    for epoch in trange(1, num_epochs + 1):
        time_start = time.time()

        pred = model(coords)
        loss = loss_fn(pred, gt)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        torch.mps.synchronize()
        total_time += time.time() - time_start
   
        if not epoch % steps_til_summary:
            with torch.no_grad():
                train_iter.append(epoch)
                train_psnr.append((psnr_fn(invnorm(model(coords)), invnorm(gt))).item())        
                
    with torch.no_grad():
        pred = invnorm(model(coords))
        
    ret_dict = {
        'train_iter': train_iter,
        'train_psnr': train_psnr,
        'pred': pred,
        'model_state': model.state_dict(),
    }
    return ret_dict