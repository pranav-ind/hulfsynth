import math
import torch
from torch import nn


norm = lambda img : (img - img.min())/(img.max() - img.min())
class ReLULayer(nn.Module):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.relu(x)
        return x

class SineLayer(nn.Module):
    """
        Implicit Neural Representations with Periodic Activation Functions
        Implementation based on https://github.com/vsitzmann/siren?tab=readme-ov-file
        - tune the parameter siren_factor
    """
    def __init__(self, in_size, out_size, siren_factor=30.0, **kwargs):
        super().__init__()
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        self.siren_factor = siren_factor
        self.linear = nn.Linear(in_size, out_size, bias=True)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sin(self.siren_factor * x)
        return x


class WIRELayer(nn.Module):
    """
        Implicit representation with Gabor nonlinearity
        Implementation based on https://github.com/vishwa91/wire
    """
    def __init__(self, in_size, out_size, wire_omega: float = 30.0, wire_sigma: float = 40.0, **kwargs):
        super().__init__()
        self.omega_0 = wire_omega  # Frequency of wavelet
        self.scale_0 = wire_sigma  # Width of wavelet
        self.freqs = nn.Linear(in_size, out_size, bias=True)
        self.scale = nn.Linear(in_size, out_size, bias=True)

    def forward(self, x):
        omega = self.omega_0 * self.freqs(x)
        scale = self.scale(x) * self.scale_0
        x = torch.cos(omega) * torch.exp(-(scale * scale))
        return x



class FourierFeatures(nn.Module):
    """ Positional encoder from Fourite Features [Tancik et al. 2020]
     Implementation based on https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb """
    def __init__(self,
                 coord_size: int,
                 freq_num: int,
                 freq_scale: float = 1.0):
        super().__init__()
        self.freq_num = freq_num  # Number of frequencies
        self.freq_scale = freq_scale  # Standard deviation of the frequencies
        self.B_gauss = torch.normal(0.0, 1.0, size=(coord_size, self.freq_num)) * self.freq_scale

        # We store the output size of the module so that the INR knows what input size to expect
        self.out_size = 2 * self.freq_num

    def forward(self, coords):
        # Map the coordinates to a higher dimensional space using the randomly initialized features
        b_gauss_pi = 2. * torch.pi * self.B_gauss.to(coords.device)
        prod = coords @ b_gauss_pi
        # Pass the features through a sine and cosine function
        out = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)
        return out


class MLP(nn.Module):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 layer_class: nn.Module = ReLULayer,
                 **kwargs):
        super().__init__()

        a = [layer_class(in_size, hidden_size, **kwargs)]
        for i in range(num_layers - 1):
            a.append(layer_class(hidden_size, hidden_size, **kwargs))
        
        
        a.append(nn.Linear(hidden_size, out_size)) #Final layer: Linear Layer without siren activation 
        self.layers = nn.ModuleList(a)        
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()  # For image intensity outputs (ensures non-negative intensity values)
        self.softmax = nn.Softmax(dim=-1) # For segmentation probability outputs
            
        

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (*chunk, in_size) where *chunk = number of sampled points and in_size = num_dims
            ex: (36864, 3)
        
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] containing:
                - output_image (torch.Tensor): Voxel intensity values (HF resolution); Shape: (*chunk,) #(one intensity per input sample)
                - output_image_pre (torch.Tensor): Raw voxel intensity values before activation; Shape: (*chunk,)
                - output_seg (torch.Tensor): Segmentation probabilities with Softmax activation; Shape: (*chunk,num_classes) where num_classes = out_size - 1
                - output_seg_pre (torch.Tensor): Raw segmentation logits before Softmax

        Notes:
            - Pre-activation outpus: output_image_pre, output_seg_pre are used for regularization
            - When using SIREN tune omega_0 parameter and run initialize_siren_weights(); SIREN will not work without proper init.
        """

        for i, layer in enumerate(self.layers):
            x = layer(x)
        # Split output layer: first neuron for image, remaining neurons for segmentation (as different activation functions are applied to intensity and segmentations)
        # output_image_pre = x[:,:4] #output image neuron before applying activation function
        output_image_pre = x[:,0] #output image neuron before applying activation function
        output_seg_pre = x[:,1:] #output seg neurons before applying activation function
        output_image = self.relu(output_image_pre)
        output_seg = self.softmax(output_seg_pre)
        
        return output_image, output_image_pre, output_seg, output_seg_pre



def initialize_activation_weights(name: str, network: MLP, SIREN_FACTOR: float, WIRE_OMEGA: float):
    if(name=='SIREN'): #name of the activation function
        initialize_siren_weights(network, SIREN_FACTOR)
    else:
        initialize_wire_weights(network, WIRE_OMEGA)


def initialize_siren_weights(network: MLP, omega: float):
    
    '''
    Notes:
    - See SIREN paper supplement Sec. 1.5 for discussion
    - To be executed along with the instance of MLP(). 
    HINT: Parse the object of MLP() as an argument to this function 
    '''

    old_weights = network.layers[1].linear.weight.clone()
    with torch.no_grad():
        # First layer initialization
        num_input = network.layers[0].linear.weight.size(-1)
        network.layers[0].linear.weight.uniform_(-1 / num_input, 1 / num_input)
        # Subsequent layer initialization uses based on omega parameter
        for layer in network.layers[1:-1]:
            num_input = layer.linear.weight.size(-1)
            layer.linear.weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
        # Final linear layer also uses initialization based on omega parameter
        num_input = network.layers[-1].weight.size(-1)
        network.layers[-1].weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
        
    # Verify that weights did indeed change
    new_weights = network.layers[1].linear.weight
    assert (old_weights - new_weights).abs().sum() > 0.0




def initialize_wire_weights(network: MLP, omega: float):
    """ See SIREN paper supplement Sec. 1.5 for discussion """
    old_weights = network.layers[1].freqs.weight.clone()
    with torch.no_grad():
        # First layer initialization
        num_input = network.layers[0].freqs.weight.size(-1)
        network.layers[0].freqs.weight.uniform_(-1 / num_input, 1 / num_input)
        network.layers[0].scale.weight.uniform_(-1 / num_input, 1 / num_input)
        # Subsequent layer initialization based on omega parameter
        for layer in network.layers[1:-1]:
            num_input = layer.freqs.weight.size(-1)
            layer.freqs.weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
            layer.scale.weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
        # Final linear layer also uses initialization based on omega parameter
        num_input = network.layers[-1].weight.size(-1)
        network.layers[-1].weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
        network.layers[-1].weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
        
    # Verify that weights did indeed change
    new_weights = network.layers[1].freqs.weight
    assert (old_weights - new_weights).abs().sum() > 0.0





#Test case:
'''
SIREN_FACTOR = 30.0 
siren_inr = MLP(in_size=3,
                out_size=5,
                hidden_size=8,
                num_layers=3,
                layer_class=SineLayer, 
                siren_factor=SIREN_FACTOR,
                )
# Re-initialize the weights and make sure they are different
initialize_siren_weights(siren_inr, SIREN_FACTOR)
'''