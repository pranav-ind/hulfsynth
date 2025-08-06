from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi
import torch.nn as nn
from kornia.losses import total_variation
from monai.losses import DiceLoss, DiceCELoss, SoftclDiceLoss, DiceFocalLoss, NACLLoss
from tqdm import tqdm, tqdm_notebook
from sklearn.metrics import mean_squared_error
from monai.losses.dice import *  # NOQA
from IPython.display import clear_output
from monai import metrics
import monai
from matplotlib import pyplot as plt
import tempfile, os
import copy
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm, tqdm_notebook




from Models.models import Siren, Finer
from Utils.utils import get_full_img, norm, get_device, dice_stack_helper, get_model, ClearCache
from Data.load_data_3d import load_data, get_gt_seg
from Utils.defaults import default_config
from Utils.plotting_utils2 import plot_seg_results_paper, plot_final_results_paper, plot_hf_results_paper
from Utils.plotting_utils import loss_plot, plot_image_metrics, plot_4_images
from LFSynth.ContrastModulation import ContrastModulation


def define_coords(img_shape) -> torch.Tensor:
    """
    defines coordinate between -1 to 1 of shape ndims
    returns tensor of shape (*img_shape, ndims)
    """
    ndims = len(img_shape)
    coords = [torch.linspace(-1, 1, img_shape[i])
              for i in range(ndims)]

    coords = torch.meshgrid(*coords, indexing=None)
    coords = torch.stack(coords, dim=ndims)

    return coords


class CoordsPatch(Dataset):
    #Adapated from INRMorph - https://github.com/aisha-lawal/inrmorph/blob/506b43125150a8671f54d9e23d3f27c3a85e43dc/data_modules/inrmorph.py
    def __init__(self, patch_size, num_patches, image):
        # super(self, CoordsPatch).__init__()
        self.patch_size = np.ceil(np.array(patch_size) / 2).astype(np.int16)
        
        self.ndims = len(self.patch_size)
        self.image = image
        self.img_shape = self.image.shape
        self.coords = define_coords(self.img_shape)
        # print(self.coords.shape)
        self.dx = torch.div(2, torch.tensor(self.coords.shape[:-1]))
        # print(self.dx)
        self.num_patches = num_patches  # how many random patches to sample
        # self.set_seed = set_seed(42)

        self.patch_size = np.ceil(np.array(patch_size) / 2).astype(np.int16)
        # print("patch size", self.patch_size)
        patch_dx_dims = torch.tensor(self.patch_size) * self.dx
        

        patch_coords = [torch.linspace(-patch_dx_dims[i], patch_dx_dims[i], 2 * self.patch_size[i]) for i in range(self.ndims)]
        # print("path coords : ", len(patch_coords), patch_coords[0].shape, patch_coords[1].shape, patch_coords[2].shape )

        patch_coords = torch.meshgrid(*patch_coords, indexing=None)
        self.patch_coords = torch.stack(patch_coords, dim=self.ndims)

        coords = self.coords[self.patch_size[0]:-self.patch_size[0],
                 self.patch_size[1]:-self.patch_size[1], self.patch_size[2]:-self.patch_size[2], ...] #need to re-check this
        # print("coords : ", coords.shape)
        self.spatial_size = coords.shape[:-1]
        # print("spatial size : ", self.spatial_size)
        self.coords = coords
        

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        
        indx = np.random.randint(0, np.prod(self.spatial_size))
        inds = np.unravel_index(indx, self.spatial_size)


        center = self.coords[inds[0], inds[1], inds[2], :]
        coords = torch.clone(self.patch_coords)
        


        coords[..., 0] = coords[..., 0] + center[0]
        coords[..., 1] = coords[..., 1] + center[1]
        coords[..., 2] = coords[..., 2] + center[2]
        return coords


#helper functions : chunk_size = (86, 96, 96) and lf = (43, 48, 96)
if __name__ == '__main__':
    patch_size = [24, 22, 24]
    IO = torch.randn(192, 172, 172)
    coord_input_loader = DataLoader(dataset=CoordsPatch(patch_size=patch_size, num_patches=120, image=IO), batch_size=1, shuffle=False, num_workers=4, drop_last=True)
    op = next(iter(coord_input_loader))
    print(type(op))