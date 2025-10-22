import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np

import random
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import copy
import pandas as pd



import lightning as pl
from lightning.pytorch.loggers import WandbLogger
import wandb
import pprint

import monai
from monai.losses import DiceLoss, DiceCELoss, SoftclDiceLoss, DiceFocalLoss, NACLLoss
from kornia.losses import total_variation
import kornia


from Models.model import MLP, initialize_siren_weights, SineLayer, ReLULayer
from Models.model_trainer import ModelTrainerModule


from Models.models import Siren, Finer

from Utils.utils import get_full_img, norm, get_device, dice_stack_helper, get_model, ClearCache
# from Data.load_ixi import load_data, load_sensitivity_data, get_hf_observed_segmentations
from Utils.defaults import default_config
from Utils.plotting_utils2 import plot_seg_results_paper, plot_final_results_paper, plot_hf_results_paper, plot_8_images_2rows, plot_3_images_row
from Utils.plotting_utils import loss_plot, plot_image_metrics, plot_4_images, plot_2_images
from LFSynth.ContrastModulation import ContrastModulation
from Data.load_ixi import get_hf_observed_segmentations as get_hf_observed_segmentations_ixi, load_sensitivity_data
from Data.load_ixi import load_data as load_ixi_data
from LFSynth.HF_ContrastEstimation import load_val_data, get_hf_observed_segmentations as get_hf_observed_segmentations_val


from Data.patchwise3D import RandomPointsDataset
from onnxruntime import InferenceSession
import onnx


def download_artifacts(run_list):
    artifact_dir_list = []

    api = wandb.Api()
    # run_id = "tmc3i3eo" #change this every time
    entity = "pi58-sussex/"
    project = "hulfsynth/"
    for run_id in run_list:
        run = api.run(entity + project + run_id)
        logged_artifacts = run.logged_artifacts(10)
        # print(logged_artifacts[0].name, logged_artifacts[1].name, logged_artifacts(temp)) #check number of logged artifacts in this list. usually should return two
        model_version = logged_artifacts[0].name
        print("Model version: ", model_version)
        artifact = api.artifact(entity + project +  model_version)
        artifact_dir = artifact.download()
        artifact_dir_list.append(artifact_dir)
    return artifact_dir_list


def get_data(config):
    ixi_nums = ["102", "105", "127", "128", "130"]
    dataset_num = config["dataset_num"]
    sens_id = config["sens_id"]
    if dataset_num in ixi_nums:
        if(str(sens_id) !="-1"):
            print("Loading sesitivity data: ", sens_id)
            hf_ground_truth, lf_gt, lf_gt_seg_dice, M = load_sensitivity_data(config)
        else:
            print("Loading IXI data: ", dataset_num)
            hf_ground_truth, lf_gt, lf_gt_seg_dice, M = load_ixi_data(config)
        (wm_obs_seg, gm_obs_seg, csf_obs_seg, bg_obs_seg) = get_hf_observed_segmentations_ixi(config["dataset_num"], config)
        config["slice"] = 175
    else:
        print("Loading val data: ", dataset_num)
        hf_ground_truth, lf_gt, lf_gt_seg_dice, M = load_val_data(target_type = 'ulf' , config = config)
        (wm_obs_seg, gm_obs_seg, csf_obs_seg, bg_obs_seg) = get_hf_observed_segmentations_val(config)
        config["slice"] = 19
        config["slice"] = 11 if str(dataset_num) == '0011' else (24 if str(dataset_num) == '0015' else 19) #slice_num and dataset_num[11: 0011, 24: 0015, 19: others]

    return (hf_ground_truth, lf_gt, lf_gt_seg_dice, M), (wm_obs_seg, gm_obs_seg, csf_obs_seg, bg_obs_seg)


def add_rician(size_lf, v=1e-3, s=1e-3):
    # v =8, s = 5
    '''
        Adding Rician Noise using the magnitude of a Bivariate Normal Distribution with non-zero mean
        Reference : https://en.wikipedia.org/wiki/Rice_distribution
    '''
    N = 1
    for i in (size_lf):
        N = N * i  #Num of samples
    noise = np.random.normal(scale=s, size=(N, 2)) + [[v,0]]
    noise = np.linalg.norm(noise, axis=1)
    return (noise.reshape(size_lf))

def contrast_modulation(pred_seg, pred_img, config):
    downsampled_points = config["downsampled_points"]
    M = config["M"]
    size_lf = config["size_lf"]
    pred_seg = torch.from_numpy(pred_seg)
    pred_img = torch.from_numpy(pred_img)
    # imgs_list = [(F.interpolate((pred_seg[i] * pred_img).permute(2,0,1).unsqueeze(0), scale_factor=0.5).squeeze(0).permute(1,2,0)) for i in range(pred_seg.shape[0])]
    imgs_list = [(F.interpolate((pred_seg[i] * pred_img).permute(2,0,1).unsqueeze(0), size=size_lf[:-1]).squeeze(0).permute(1,2,0)) for i in range(pred_seg.shape[0])]
    bg_img = (imgs_list[0]).reshape(size_lf)
    wm_img = (imgs_list[1]).reshape(size_lf) * M[0]
    gm_img = (imgs_list[2]).reshape(size_lf) * M[1]
    csf_img = (imgs_list[3]).reshape(size_lf) * M[2]

    lf_img = wm_img + gm_img + csf_img #+ bg_img
    rician_noise = torch.from_numpy(add_rician(lf_img.shape)) #adding rician noise
    mask = torch.where(lf_img>0 ,1.0, 0.0)
    lf_img += (rician_noise * mask) #adding noise only to foreground voxels
    return lf_img.numpy()



def get_canny(config, img):
    #expected img shape: (H, W, D); returns (H, W, D)
    canny = kornia.filters.Canny()
    img = img[:,:,config["slice"]] #getting a 2D slice
    img_magnitude, img_canny = canny(img.unsqueeze(0).unsqueeze(0))
    return img_canny.squeeze(0).squeeze(0)

def get_interpolated(img, scale_factor=1, mode='bicubic'):
    # expected img_shape: (H, W, D)
    img = img.permute(2,0,1).unsqueeze(0) #shape: (B, D(actuall C), H, W) where D is treated as C
    trilinear = torch.nn.functional.interpolate(img, scale_factor=scale_factor, mode=mode, recompute_scale_factor=True)
    trilinear = trilinear.squeeze(0).permute(1,2,0) #shape: (H,W,D)
    return trilinear
    

class GridFitting():
    def __init__(self) -> None:
        pass
    def create_3d_coordinate_grid(self, depth, height, width, device='cpu'):
        """
        Create a 3D grid of coordinates for the entire volume
        
        Args:
            depth: Volume depth (Z dimension)
            height: Volume height (Y dimension)  
            width: Volume width (X dimension)
            device: torch device
        
        Returns:
            coords: (D, H, W, 3) tensor of coordinates in [-1, 1] range
                    coords[d, h, w] = [x, y, z] coordinate for voxel (d, h, w)
        """

        
        # Create voxel coordinates for each dimension
        z_coords = torch.linspace(0, depth - 1, depth, device=device)   # [0, D-1]
        y_coords = torch.linspace(0, height - 1, height, device=device) # [0, H-1]
        x_coords = torch.linspace(0, width - 1, width, device=device)   # [0, W-1]

        
        # Create 3D meshgrid
        # indexing='ij' means first index is Z, second is Y, third is X
        grid_z, grid_y, grid_x = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        
        # Normalize to [0, 1]
        grid_x_norm = grid_x / (width - 1)
        grid_y_norm = grid_y / (height - 1)
        grid_z_norm = grid_z / (depth - 1)
        
        # Convert to grid_sample format [-1, 1]
        grid_x_gs = grid_x_norm * 2 - 1
        grid_y_gs = grid_y_norm * 2 - 1
        grid_z_gs = grid_z_norm * 2 - 1

    
        # Stack into (D, H, W, 3) where last dim is [x, y, z]
        # # CRITICAL: grid_sample expects [x, y, z] order!
        coords = torch.stack([grid_x_gs, grid_y_gs, grid_z_gs], dim=-1)
        
        return coords


    def sample_volume_at_coordinates(self, volume, coords):
        """
        Sample 3D volume at given coordinates using grid_sample
        
        Args:
            volume: (1, C, D, H, W) tensor - the volume to sample from
            coords: (D, H, W, 3) tensor - coordinates to sample at
        
        Returns:
            sampled: (1, C, D, H, W) tensor - sampled volume
        """
        
        # grid_sample expects grid shape: (N, D, H, W, 3)
        # We have (D, H, W, 3), so add batch dimension
        grid = coords.unsqueeze(0)  # (1, D, H, W, 3)
        
        
        sampled = F.grid_sample(
            volume,
            grid,
            mode='bilinear',  # Actually trilinear for 3D!
            padding_mode='border',
            align_corners=True
        )
        
        
        return sampled



    def fit(self, volume, volume_size=None):
        """
        input shape: (H, W, D)
        output shape: (H, W, D)
        """
        volume = volume.permute(2,0,1) #changing shape to (D, H, W)
        if(volume_size!= None):
            D, H, W = volume_size
        else:
            D, H, W = volume_size = volume.shape
        
        
        volume = volume.unsqueeze(0).unsqueeze(0) #changing shape to (N, C, D, H, W)
        coords = self.create_3d_coordinate_grid(D, H, W) #(1, D,H,W,3)
        reconstructed = self.sample_volume_at_coordinates(volume, coords)
        volume = volume.squeeze(0).squeeze(0).permute(1,2,0) #changing shape to (H, W, D)
        reconstructed = reconstructed.squeeze(0).squeeze(0).permute(1,2,0) #changing shape to (H, W, D)
        return reconstructed
voxelgrid = GridFitting()


from typing import Tuple
def get_session(model_loc):
    # model_loc = './artifacts/tmc3i3eo:v0/model.onnx'
    onnx_model = onnx.load(model_loc)
    onnx.checker.check_model(onnx_model)
    # print(onnx_model.graph.input)
    onnx_model.graph.input[0].type.tensor_type.shape.dim[0].ClearField('dim_value')
    onnx_model.graph.input[0].type.tensor_type.shape.dim[1].ClearField('dim_value')
    sess = InferenceSession(onnx_model.SerializeToString())
    return sess

def get_coords(resolution: Tuple[int, ...]):
    
    print("sampling at resolution: ", resolution)
    meshgrid = torch.meshgrid([torch.arange(0, i, device=device) for i in resolution], indexing='ij')
    coords = torch.stack(meshgrid, dim=-1)
    coords_norm = coords / torch.tensor(resolution, device=device) * 2 - 1
    coords_norm_ = coords_norm.reshape(-1, coords.shape[-1])
    coords_norm_ += (torch.randn_like(coords_norm_) * 0.001) #adding gaussian noise with std = 0.001
    return coords_norm_


def sample_at_resolution(sess, hf_ground_truth, config):
    """ Evaluate our INR on a grid of coordinates in order to obtain an image. """
    # sess = InferenceSession(onnx_model.SerializeToString())
    resolution = hf_ground_truth.shape
    predictions_, _, pred_seg_, _ = sess.run(None, {"onnx::MatMul_0" : (get_coords(resolution).cpu().numpy())})
    # predictions_, _, pred_seg_, _ = self.forward(ff_pos_enc(coords_norm_ + (torch.randn_like(coords_norm_) * 0.001))) #adding gaussian noise with std = 0.01    
    resolution_seg = list(resolution) + [pred_seg_.shape[-1]] #adding num_tissues to the resolution shape
    predictions_hf = predictions_.reshape(resolution)
    pred_seg_ = pred_seg_.reshape(resolution_seg)
    pred_seg = [pred_seg_[:,:,:,i].reshape(resolution) for i in range(pred_seg_.shape[-1])]
    pred_seg = np.stack(pred_seg, axis = 0)
    pred_lf = contrast_modulation(pred_seg, predictions_hf, config)
    return predictions_hf, pred_seg, pred_lf



from Utils.utils import MSLC, piqe_score
from deepinv.loss.metric import LPIPS, HaarPSI
def image_metrics(config, obs_img, pred_img):
    #Expected shapes: 3D (H, W, D)
    middle_slices = 5
    slice_num = config["slice"]
    psnr_value = monai.metrics.PSNRMetric(max_val = 1.0) #expects shape: BCHWD
    ssim_value = monai.metrics.regression.SSIMMetric(spatial_dims=3, data_range=1.0) #expects shape: BCHWD
    lpips = LPIPS()
    haar = HaarPSI(norm_inputs="clip")
    mslc = MSLC()
    # piqe_score()
    psnr_ =  psnr_value(pred_img.unsqueeze(0).unsqueeze(0), obs_img.unsqueeze(0).unsqueeze(0))
    ssim_ =  ssim_value(pred_img.unsqueeze(0).unsqueeze(0), obs_img.unsqueeze(0).unsqueeze(0))
    lpips_ = lpips(pred_img.unsqueeze(0).permute(3,0,1,2)[slice_num-middle_slices:slice_num+middle_slices], obs_img.unsqueeze(0).permute(3,0,1,2)[slice_num-middle_slices:slice_num+middle_slices]).mean() #computing score for middle 10 slices
    haar_ = haar(pred_img.unsqueeze(0).permute(3,0,1,2)[slice_num-middle_slices:slice_num+middle_slices], obs_img.unsqueeze(0).permute(3,0,1,2)[slice_num-middle_slices:slice_num+middle_slices]).mean() #computing score for middle 10 slices

    mslc_ = torch.tensor([mslc(pred_img[:,:,i].cpu().numpy()) for i in range(slice_num-5, slice_num+5)]).mean() #computing MSLC scor for middle 6 slices
    piqe_ = torch.tensor([piqe_score(pred_img[:,:,i].cpu().numpy()) for i in range(slice_num-5, slice_num+5)]).mean() #computing PIQE scor for middle 10 slices
    return psnr_, haar_, ssim_, lpips_, mslc_, piqe_
# psnr_, haar_, ssim_, lpips_, mslc_, piqe_ = image_metrics(config, gt_image[:,:,:,0], final_img) #Test function



run_list = ['0ye4p313', 'yxi36ok6', '25gvh7d5', 'y9stwkw1', 'evp37xzg', 'a6imu6de', 'wvdtfihb', 'r8xf31s9']
dataset_list = ['102']
sens_id_list = [1, 2, 3, 4, 5, 6, 7, 8]

artifact_dir_list = download_artifacts(run_list)
hf_obs_list = []
lf_obs_list = []

lf_obs_seg_list = []
hf_obs_seg_list = []

hf_pred_list = []
lf_pred_list = []

lf_pred_seg_list = []
hf_pred_seg_list = []

config_list = []
final_img_list = []


device = get_device()
for i in range(len(artifact_dir_list)):
    model_loc = artifact_dir_list[i] + '/model.onnx'
    config = copy.deepcopy(default_config)
    config["dataset_num"] = dataset_list[0] #only 102
    config["sens_id"] = i + 1 #i: 0 to 7 and sens_id: 1 to 8
    (hf_ground_truth, lf_gt, lf_gt_seg_dice, M), (wm_obs_seg, gm_obs_seg, csf_obs_seg, bg_obs_seg) = get_data(config)
    gt_image = torch.tensor(norm(hf_ground_truth)).unsqueeze(-1)
    gt_image = gt_image.to(torch.float32)
    lf_gt = torch.tensor(norm(lf_gt)).unsqueeze(-1)
    lf_gt = lf_gt.to(torch.float32)
    config["size"] = hf_ground_truth.shape
    config["size_lf"] = lf_gt.shape[:-1]
    config["M"] = M
    hf_observed_seg_dice = torch.stack((bg_obs_seg, wm_obs_seg, gm_obs_seg, csf_obs_seg), dim=0).unsqueeze(0)
    slice_num = config["slice"]
    inference_session = get_session(model_loc)
    pred_hf, pred_hf_seg, pred_lf = sample_at_resolution(inference_session, hf_ground_truth, config)
    pred_hf, pred_hf_seg, pred_lf = torch.from_numpy(pred_hf), torch.from_numpy(pred_hf_seg), torch.from_numpy(pred_lf)  

    pred_lf_seg = [(F.interpolate(pred_hf_seg[i].permute(2,0,1).unsqueeze(0), size=lf_gt.shape[:-2]).squeeze(0)).permute(1,2,0).reshape(lf_gt_seg_dice.shape[2:]) for i in range(pred_hf_seg.shape[0])] #downsampling pred_seg
    pred_lf_seg = torch.stack(pred_lf_seg,axis = 0) # shape(4 *H_lf, *W_lf, *D)
    pred_hf_seg = [pred_hf_seg[i].reshape(config["size"]) for i in range(pred_hf_seg.shape[0])]
    pred_hf_seg = torch.stack(pred_hf_seg, axis=0)
    final_img = norm((pred_hf * pred_hf_seg[1]) + (pred_hf * pred_hf_seg[2]) + (pred_hf * pred_hf_seg[3]))

    hf_obs_list.append(gt_image[:,:,:,0])
    lf_obs_list.append(lf_gt[:,:,:,0])
    hf_obs_seg_list.append(hf_observed_seg_dice[0])
    lf_obs_seg_list.append(lf_gt_seg_dice[0])
    hf_pred_list.append(pred_hf.reshape(config["size"]))
    lf_pred_list.append(pred_lf.reshape(config["size_lf"]))
    hf_pred_seg_list.append(hf_observed_seg_dice[0])
    lf_pred_seg_list.append(lf_gt_seg_dice[0])
    final_img_list.append(final_img)
    config_list.append(config)


#plots
for i in range(len(dataset_list)):
    single_subject = [lf_obs_list[i], lf_pred_list[i], hf_obs_list[i], final_img_list[i]]
    interpolated = [get_interpolated(lf_obs_list[i], scale_factor=1, mode='bilinear') , get_interpolated(lf_obs_list[i], scale_factor=1, mode='bicubic'), voxelgrid.fit(lf_obs_list[i])]
    single_subject = single_subject + interpolated
    canny_subject = [get_canny(config_list[i], single_subject[j]) for j in range(len(single_subject))]
    titles = ["Y", "$\\hat{Y}$", "X", "$\\hat{X}_{w}$", "bilinear", "bicubibc", "Voxel Grid"] 
    fig = plot_images_in_row(config_list[i], single_subject, canny_subject, titles, cmap= 'gray', figsize=(11,4))
    fig.savefig('./sensitivity_results/' + sens_id_list[i] + '.pdf')
