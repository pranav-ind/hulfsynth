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
from IPython.display import clear_output
import wandb
import random


from Models.models import Siren, Finer
from Utils.utils import get_full_img, norm, get_device, dice_stack_helper, get_model, ClearCache
from Data.load_data_3d import load_data, get_gt_seg
from Utils.defaults import default_config
from Utils.plotting_utils2 import plot_seg_results_paper, plot_final_results_paper, plot_hf_results_paper
from Utils.plotting_utils import loss_plot, plot_image_metrics, plot_4_images
from LFSynth.ContrastModulation import ContrastModulation
from test3D import ModelTrainer

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


def visualize_volume_slices(volume1, volume2, axis=0, num_slices=16, title1='Volume 1', title2='Volume 2'):
    """
    Visualizes `num_slices` slices from two 3D volumes side-by-side.
    
    Args:
        volume1 (torch.Tensor): 3D tensor (D, H, W)
        volume2 (torch.Tensor): 3D tensor (D, H, W)
        axis (int): Axis along which to slice (0=z, 1=y, 2=x)
        num_slices (int): Number of slices to show (default=16)
        title1 (str): Title for first volume
        title2 (str): Title for second volume
    """

    # assert volume1.shape == volume2.shape, "Volumes must be the same shape"
    assert volume1.dim() == 3, "Volumes must be 3D tensors"

    # Select slice indices (evenly spaced)
    dim = volume1.shape[axis]
    step = max(dim // num_slices, 1)
    slice_indices = list(range(0, dim, step))[:num_slices]

    fig, axes = plt.subplots(4, num_slices // 2, figsize=(16, 8))
    axes = axes.flatten()

    for i, idx in enumerate(slice_indices):
        if axis == 0:
            slice1 = volume1[idx, :, :].cpu().numpy()
            slice2 = volume2[idx, :, :].cpu().numpy()
        elif axis == 1:
            slice1 = volume1[:, idx, :].cpu().numpy()
            slice2 = volume2[:, idx, :].cpu().numpy()
        elif axis == 2:
            slice1 = volume1[:, :, idx].cpu().numpy()
            slice2 = volume2[:, :, idx].cpu().numpy()
        else:
            raise ValueError("Invalid axis")

        axes[i].imshow(slice1, cmap='gray')
        axes[i].set_title(f'{title1} - Slice {idx}')
        axes[i].axis('off')

        # Add the second volume slice in the next row (offset by num_slices)
        axes[i + num_slices].imshow(slice2, cmap='gray')
        axes[i + num_slices].set_title(f'{title2} - Slice {idx}')
        axes[i + num_slices].axis('off')

    plt.tight_layout()
    # plt.show()
    return fig


#helper functions : chunk_size = (86, 96, 96) and lf = (43, 48, 96)
if __name__ == '__main__':
    patch_size = [48, 44, 48]
    IO = torch.randn(192, 172, 172)
    coord_input_loader = DataLoader(dataset=CoordsPatch(patch_size=patch_size, num_patches=100, image=IO), batch_size=1, shuffle=False, num_workers=4, drop_last=True)
    # op = next(iter(coord_input_loader))
    # print(type(op))
    device = 'cpu'
    config = copy.deepcopy(default_config)
    hf_ground_truth, lf_gt, prior_seg_dice, lf_gt_seg_dice, M = load_data(1, config) #uncomment
    target_gt = F.pad(torch.from_numpy(lf_gt).to(torch.float32), (0, 0, 0, 0, 0, 1)).to(device).permute(2,0,1) #shape : [192, 88, 96]
    target_seg = F.pad(lf_gt_seg_dice.to(torch.float32), (0, 0, 0, 0, 0, 1)).to(device).permute(0,1,4,2,3) #shape : [1, 4, 192, 88, 96]
    target_hf = torch.from_numpy(norm(hf_ground_truth))[:,1:-1,:].to(device).float() #shape: [192, 172, 192]

    project_ = "hulfsynth_ulfenc"
    # run_name = "run_" + str(run_id)
    wandb.login()
    run = wandb.init(project=project_)

    # print(target_gt.shape, target_seg.shape)
    with run:
        for idx, patch_grid in (enumerate(coord_input_loader)):
            # coord_chunk = patch_grid #shape(1, 48, 43, 48, 3)
            patch_grid = patch_grid.to(device)
            target_patch = F.grid_sample(target_gt.unsqueeze(0).unsqueeze(0), patch_grid, mode='bilinear')[:,:,:,::2,::2] #output : (1,1, 48,22,24) #for 5D-input, bilinear == trilinear 
            target_seg_patch = F.grid_sample(target_seg, patch_grid, mode='bilinear')[:,:,:,::2,::2] #output : (1,4, 48,22,24)
            target_hf_patch = F.grid_sample(target_hf.unsqueeze(0).unsqueeze(0), patch_grid, mode='bilinear') #output : (1,1, 48,43,48)
            print("ID: ", idx, target_patch.shape, target_seg_patch.shape)
            if(idx%50==0):
                clear_output(wait=True)
            fig = visualize_volume_slices(target_patch.squeeze(0).squeeze(0), target_seg_patch.squeeze(0)[2], axis=0, num_slices=20, title1='gt', title2='seg_gt')
            # plt.show()
            fig.savefig('/Users/pi58/Library/CloudStorage/Box-Box/PhD/MPhil/Projects/Hulf_Synth/temp_patches/'+ str(idx) + '.png')

            # path_to_obj = "../input/kernel-files/wolf.obj"
            # Initialize run
            # wandb.init()

            # wandb.log({"3d_object":wandb.Object3D(target_patch.squeeze(0).squeeze(0).detach().cpu().numpy().reshape(-1,3))})  # Log 3D object
            n_slice = random.randint(1, 48)
            # wandb.log({
            #     "target_patch": wandb.Image(target_patch.squeeze(0).squeeze(0)[n_slice].cpu().numpy() , mode='L'), #random slice=5
            #     "seg_patch": wandb.Image(target_seg_patch.squeeze(0)[2,n_slice].cpu().numpy(), mode='L') #random slice=5,
            # })

    # trainer = ModelTrainer(config, lf_gt, prior_seg_dice, lf_gt_seg_dice, M) #init
    # model, losses = (trainer.train_inr())
    # model_saving_path =  "./temp_model.onnx"
    # torch.onnx.export(model, trainer.coord_chunk, model_saving_path, dynamo=True)
    # print("locally saved model to: ", model_saving_path)
    # wandb.save(model_saving_path)

    # run.log_model(path=model_saving_path, name="model")
    # run.finish()



