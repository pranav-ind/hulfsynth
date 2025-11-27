import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from typing import Any, List
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
from pypiqe import piqe
import matplotlib.pyplot as plt
import time
import monai
import monai
import pandas as pd
from fsl.wrappers.fast import fast as fast
# from Models.models import Siren, Finer
# import Models.models as models


#Helper Functions
norm = lambda img : (img - img.min())/(img.max() - img.min()) #normalizes an image
dice_stack_helper = lambda img : torch.stack((img[:,:,3].reshape(size), img[:,:,0].reshape(size), img[:,:,1].reshape(size), img[:,:,2].reshape(size)), dim=0).unsqueeze(0) #Expected shape = (1,*hf_spatial,4)

# def get_model(config):
#     if(config["model"] == 0):
#         return models.Siren(in_features=config["in_features"], out_features=8, hidden_features=config["hidden_features"], hidden_layers=config["hidden_layers"], outermost_linear=True, first_omega_0 = config["w0"], hidden_omega_0 = config["w0"])
#     else:
#         return models.Finer(in_features=config["in_features"], out_features=8, hidden_layers=config["hidden_layers"], hidden_features=config["hidden_features"], first_omega=config["w0"], hidden_omega=config["w0"], init_method='sine', init_gain=1, fbs=None)


def psnr(pred, ref):
    max_value = ref.max()
    mse = torch.mean((pred - ref) ** 2, dim=(-2, -1))
    out = 20 * torch.log10(max_value / torch.sqrt(mse))
    return out.mean()

def normalize_psnr(psnr_value, min_val=15.0, max_val=30.0):
    psnr_norm = (psnr_value - min_val) / (max_val - min_val)
    return torch.clamp(psnr_norm, 0.0, 1.0)

def get_device():
    
    if(torch.cuda.is_available()):
        device = torch.device("cuda:0")
    elif(torch.backends.mps.is_available()):
        device = torch.device("mps")
    else:
        print("MPS or CUDA Unavailable. Using CPU")
        device = torch.device("cpu")
    
    # device = torch.device("cpu")
    print("Device = ", device)

    return device


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad



def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)





def get_metrics(model_output_img, model_output_seg , hf_ground_truth):
    # High-Field Reconstruction Metrics
    metrics = {"psnr" : 0, "ssim": 0, "mse": 0}

    hf_output = model_output_img.clip(min=0).cpu().detach()
    model_output_seg_new = model_output_seg.to(torch.float32).cpu().detach()
    hf_result = (hf_output[:,:,0] * model_output_seg_new[:,:,0]) + (hf_output[:,:,1] * model_output_seg_new[:,:,1]) + (hf_output[:,:,2] * model_output_seg_new[:,:,2]) +  (hf_output[:,:,3] * model_output_seg_new[:,:,3])
    hf_pred = hf_result.view(hf_ground_truth.shape)
    
    hf_gt_normalized = torch.from_numpy(norm(hf_ground_truth))
    hf_pred_normalized = norm(hf_pred)

    psnr = monai.metrics.PSNRMetric(max_val = 1.0)
    ssim = monai.metrics.regression.SSIMMetric(spatial_dims=2, data_range=1.0)
    mse = monai.metrics.MSEMetric()

    metrics["psnr"] = psnr(hf_gt_normalized.unsqueeze(0), hf_pred_normalized.unsqueeze(0)).item()
    metrics["ssim"] = ssim(hf_gt_normalized.unsqueeze(0).unsqueeze(0), hf_pred_normalized.unsqueeze(0).unsqueeze(0)).item()
    metrics["mse"] = mse(hf_gt_normalized.unsqueeze(0).unsqueeze(0), hf_pred_normalized.unsqueeze(0).unsqueeze(0)).item()
    return metrics



def get_full_img(seg, img, config):
    #Takes seg and image of size [1,*spatial,4] and config dictionary
    size = config["size"]
    size_lf = config["size_lf"]
    return ((seg[:,:,3] * img[:,:,3]).reshape(size)) +  ((seg[:,:,0] * img[:,:,0]).reshape(size)) + ((seg[:,:,1] * img[:,:,1]).reshape(size)) + ((seg[:,:,2] * img[:,:,2]).reshape(size))




def pad_chunk(img, chunk_size):
    # expected input shape : (1, 43*48*48) ; expected output shape : (1, 44, 48, 48)
    # img = img.reshape(43,48,48)
    img = img.reshape(chunk_size)
    padding = (0, 0, 0, 0, 0, 1) # (padding_left, padding_right, padding_top, padding_bottom)
    padded_img = F.pad(img, padding)
    return padded_img




def lf_seg_loss(model_output_seg, chunk_size):
    lf_pred_seg_dice = torch.stack((
    pad_chunk(model_output_seg[:,:,3], chunk_size)[::2, ::2], 
    pad_chunk(model_output_seg[:,:,0], chunk_size)[::2, ::2], 
    pad_chunk(model_output_seg[:,:,1], chunk_size)[::2, ::2], 
    pad_chunk(model_output_seg[:,:,2], chunk_size)[::2, ::2]), dim=0).unsqueeze(0)

    return lf_pred_seg_dice

def model_seg_loss(model_output_seg, chunk_size):
    return torch.stack((
    pad_chunk(model_output_seg[:,:,3], chunk_size),
    pad_chunk(model_output_seg[:,:,0], chunk_size),
    pad_chunk(model_output_seg[:,:,1], chunk_size),
    pad_chunk(model_output_seg[:,:,2], chunk_size)), dim=0).unsqueeze(0)

class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()


def interpolation_metrics(interpolate_res, hf_ground_truth):
    psnr = monai.metrics.PSNRMetric(max_val = 1.0)
    ssim = monai.metrics.regression.SSIMMetric(spatial_dims=2, data_range=1.0)
    mse = monai.metrics.MSEMetric()
    
    
    metrics = {"psnr" : 0, "ssim": 0, "mse": 0}
    hf_pred = interpolate_res

    hf_gt_normalized = torch.from_numpy(norm(hf_ground_truth))
    hf_pred_normalized = norm(interpolate_res)


    metrics["psnr"] = psnr(hf_gt_normalized.unsqueeze(0).unsqueeze(0), hf_pred_normalized).item()
    metrics["ssim"] = ssim(hf_gt_normalized.unsqueeze(0).unsqueeze(0), hf_pred_normalized).item()
    metrics["mse"] = mse(hf_gt_normalized.unsqueeze(0).unsqueeze(0), hf_pred_normalized).item()

    return metrics


def segment(img_location, output_folder, t = 1, n_classes = 3, g = True, B = True, b=True):
    '''
    img_location : The location to the image that needs to be segmented. (Expecting a HF image)
    output_folder : The folder in which all the outputs of FSL FAST are expected to be stored. (Suggested to use 1 folder per dataset)
    -n,--class	number of tissue-type classes; default=3
    -t,--type	type of image 1=T1, 2=T2, 3=PD; default=T1
    -g,--segments	outputs a separate binary image for each tissue type
    -b		output estimated bias field
	-B		output bias-corrected image
    Reference : For further options : Refer https://fsl.fmrib.ox.ac.uk/fsl/docs/#/structural/fast or Use Command Line : fast 
    '''
    fast(img_location, out= output_folder+'fast', g= g, B=B, n_classes=n_classes, t=t)
    return "Segmentation stored in " + output_folder

def piqe_score(img):
    # expecting a 2D image 
    score, activityMask, noticeableArtifactMask, noiseMask = piqe(img)
    return score

class MSLC():
    '''
    Implementation: https://github.com/Bayer-Group/mr-image-metrics/tree/main
    Reference: https://www.nature.com/articles/s41598-025-87358-0#data-availability
    #TODO: This implementation is in Numpy. Modify it to PyTorch. 
    Note: This is costly as the torch image tensor will be copied to cpu and converted to numpy
    '''
    def __init__(self) -> None:
        pass
    def corr(self, x: np.ndarray, y: np.ndarray) -> float:
        if (x == y).all():
            return 1.0
        elif np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        else:
            return np.corrcoef(x, y)[0, 1]


    def get_corrcoefs(self, image: np.ndarray, distance: int = 1) -> List[float]:
        ces = []
        for x in range(0, image.shape[0] - distance):
            ces.append(self.corr(image[x, :], image[x + distance, :]))
        for y in range(0, image.shape[1] - distance):
            ces.append(self.corr(image[:, y], image[:, y + distance]))
        return ces

        """
        Parameters:
        -----------
        image: np.array (H, W)
            Reference image
        """

    def forward(self, image: np.ndarray, **kwargs: Any) -> float:
            
        ces = self.get_corrcoefs(image, distance=image.shape[1] // 2)
        return np.array(ces).mean()
    
    def __call__(self, image: np.ndarray):
        return self.forward(image)
    

#unused functions

def get_quant_results(config, lf_gt1,lf_gt2, hf_ground_truth1, hf_ground_truth2,  hf_gt_seg1, hf_gt_seg2, model_output_seg1, model_output_seg2, model_output_img1, model_output_img2):
    size = config["size"]
    size_lf = config["size_lf"]
    
    dice_score = monai.metrics.DiceMetric()
    iou_score = monai.metrics.MeanIoU()
    psnr = monai.metrics.PSNRMetric(max_val = 1.0)
    ssim = monai.metrics.regression.SSIMMetric(spatial_dims=2, data_range=1.0)
    mse = monai.metrics.MSEMetric()
    dice_stack_helper = lambda img : torch.stack((img[:,:,3].reshape(size), img[:,:,0].reshape(size), img[:,:,1].reshape(size), img[:,:,2].reshape(size)), dim=0).unsqueeze(0) #Expected shape = (1,*hf_spatial,4)



    interpolated_bicubic = [torch.nn.functional.interpolate(lf_gt1.view(size_lf).unsqueeze(0).unsqueeze(0), size=None, scale_factor=2, mode='bicubic', align_corners=None, recompute_scale_factor=True, antialias=False),
    torch.nn.functional.interpolate(lf_gt2.view(size_lf).unsqueeze(0).unsqueeze(0), size=None, scale_factor=2, mode='bicubic', align_corners=None, recompute_scale_factor=True, antialias=False)]

    interpolated_bilinear = [torch.nn.functional.interpolate(lf_gt1.view(size_lf).unsqueeze(0).unsqueeze(0), size=None, scale_factor=2, mode='bilinear', align_corners=None, recompute_scale_factor=True, antialias=False),
    torch.nn.functional.interpolate(lf_gt2.view(size_lf).unsqueeze(0).unsqueeze(0), size=None, scale_factor=2, mode='bilinear', align_corners=None, recompute_scale_factor=True, antialias=False)]

    dice_s1_siren = dice_score(hf_gt_seg1, dice_stack_helper(model_output_seg1)).mean().item()
    iou_s1_siren = iou_score(hf_gt_seg1, dice_stack_helper(model_output_seg1)).mean().item()
    bilinear_metrics_s1 = interpolation_metrics(interpolated_bilinear[0], hf_ground_truth1) #returns dict 'psnr', 'ssim, 'mse
    bicubic_metrics_s1 = interpolation_metrics(interpolated_bicubic[0], hf_ground_truth1)
    siren_metrics_s1 = interpolation_metrics(get_full_img(model_output_seg1, model_output_img1, config).unsqueeze(0).unsqueeze(0).to('cpu'), hf_ground_truth1)


    dice_s2_siren = dice_score(hf_gt_seg2, dice_stack_helper(model_output_seg2)).mean().item()
    iou_s2_siren = iou_score(hf_gt_seg2, dice_stack_helper(model_output_seg2)).mean().item()
    bilinear_metrics_s2 = interpolation_metrics(interpolated_bilinear[1], hf_ground_truth2) #returns dict 'psnr', 'ssim, 'mse
    bicubic_metrics_s2 = interpolation_metrics(interpolated_bicubic[1], hf_ground_truth2)
    siren_metrics_s2 = interpolation_metrics(get_full_img(model_output_seg2, model_output_img2, config).unsqueeze(0).unsqueeze(0).to('cpu'), hf_ground_truth2)


    df = pd.DataFrame(columns= ["model_type", "Dice1", "IoU1", "PSNR1", "SSIM1", "MSE1", "Dice2", "IoU2", "PSNR2", "SSIM2", "MSE2"])

    df["model_type"] = ["SIREN", "BICUBIC", "BILINEAR"]

    df["PSNR1"] = [siren_metrics_s1["psnr"], bicubic_metrics_s1["psnr"], bilinear_metrics_s1["psnr"]]
    df["PSNR2"] = [siren_metrics_s2["psnr"], bicubic_metrics_s2["psnr"], bilinear_metrics_s2["psnr"]]

    df["SSIM1"] = [siren_metrics_s1["ssim"], bicubic_metrics_s1["ssim"], bilinear_metrics_s1["ssim"]]
    df["SSIM2"] = [siren_metrics_s2["ssim"], bicubic_metrics_s2["ssim"], bilinear_metrics_s2["ssim"]]

    df["MSE1"] = [siren_metrics_s1["mse"], bicubic_metrics_s1["mse"], bilinear_metrics_s1["mse"]]
    df["MSE2"] = [siren_metrics_s2["mse"], bicubic_metrics_s2["mse"], bilinear_metrics_s2["mse"]]

    df["Dice1"] = [dice_s1_siren, "NA", "NA"]
    df["Dice2"] = [dice_s2_siren, "NA", "NA"]

    df["IoU1"] = [iou_s1_siren, "NA", "NA"]
    df["IoU2"] = [iou_s2_siren, "NA", "NA"]

    # print(df)
    return df



def get_quant_results2(config, lf_gt1,lf_gt2, hf_ground_truth1, hf_ground_truth2,  hf_gt_seg1, hf_gt_seg2, model_output_seg1, model_output_seg2, hf_pred1, hf_pred2):
    size = config["size"]
    size_lf = config["size_lf"]
    
    dice_score = monai.metrics.DiceMetric()
    iou_score = monai.metrics.MeanIoU()
    psnr = monai.metrics.PSNRMetric(max_val = 1.0)
    ssim = monai.metrics.regression.SSIMMetric(spatial_dims=2, data_range=1.0)
    mse = monai.metrics.MSEMetric()
    dice_stack_helper = lambda img : torch.stack((img[:,:,3].reshape(size), img[:,:,0].reshape(size), img[:,:,1].reshape(size), img[:,:,2].reshape(size)), dim=0).unsqueeze(0) #Expected shape = (1,*hf_spatial,4)



    interpolated_bicubic = [torch.nn.functional.interpolate(lf_gt1.view(size_lf).unsqueeze(0).unsqueeze(0), size=None, scale_factor=2, mode='bicubic', align_corners=None, recompute_scale_factor=True, antialias=False),
    torch.nn.functional.interpolate(lf_gt2.view(size_lf).unsqueeze(0).unsqueeze(0), size=None, scale_factor=2, mode='bicubic', align_corners=None, recompute_scale_factor=True, antialias=False)]

    interpolated_bilinear = [torch.nn.functional.interpolate(lf_gt1.view(size_lf).unsqueeze(0).unsqueeze(0), size=None, scale_factor=2, mode='bilinear', align_corners=None, recompute_scale_factor=True, antialias=False),
    torch.nn.functional.interpolate(lf_gt2.view(size_lf).unsqueeze(0).unsqueeze(0), size=None, scale_factor=2, mode='bilinear', align_corners=None, recompute_scale_factor=True, antialias=False)]

    dice_s1_siren = dice_score(hf_gt_seg1, dice_stack_helper(model_output_seg1)).mean().item()
    iou_s1_siren = iou_score(hf_gt_seg1, dice_stack_helper(model_output_seg1)).mean().item()
    bilinear_metrics_s1 = interpolation_metrics(interpolated_bilinear[0], hf_ground_truth1) #returns dict 'psnr', 'ssim, 'mse
    bicubic_metrics_s1 = interpolation_metrics(interpolated_bicubic[0], hf_ground_truth1)
    siren_metrics_s1 = interpolation_metrics(hf_pred1.unsqueeze(0).unsqueeze(0).to('cpu'), hf_ground_truth1)


    dice_s2_siren = dice_score(hf_gt_seg2, dice_stack_helper(model_output_seg2)).mean().item()
    iou_s2_siren = iou_score(hf_gt_seg2, dice_stack_helper(model_output_seg2)).mean().item()
    bilinear_metrics_s2 = interpolation_metrics(interpolated_bilinear[1], hf_ground_truth2) #returns dict 'psnr', 'ssim, 'mse
    bicubic_metrics_s2 = interpolation_metrics(interpolated_bicubic[1], hf_ground_truth2)
    siren_metrics_s2 = interpolation_metrics(hf_pred2.unsqueeze(0).unsqueeze(0).to('cpu'), hf_ground_truth2)




    df = pd.DataFrame(columns= ["model_type", "Dice1", "IoU1", "PSNR1", "SSIM1", "MSE1", "Dice2", "IoU2", "PSNR2", "SSIM2", "MSE2"])

    df["model_type"] = ["SIREN", "BICUBIC", "BILINEAR"]

    df["PSNR1"] = [siren_metrics_s1["psnr"], bicubic_metrics_s1["psnr"], bilinear_metrics_s1["psnr"]]
    df["PSNR2"] = [siren_metrics_s2["psnr"], bicubic_metrics_s2["psnr"], bilinear_metrics_s2["psnr"]]

    df["SSIM1"] = [siren_metrics_s1["ssim"], bicubic_metrics_s1["ssim"], bilinear_metrics_s1["ssim"]]
    df["SSIM2"] = [siren_metrics_s2["ssim"], bicubic_metrics_s2["ssim"], bilinear_metrics_s2["ssim"]]

    df["MSE1"] = [siren_metrics_s1["mse"], bicubic_metrics_s1["mse"], bilinear_metrics_s1["mse"]]
    df["MSE2"] = [siren_metrics_s2["mse"], bicubic_metrics_s2["mse"], bilinear_metrics_s2["mse"]]

    df["Dice1"] = [dice_s1_siren, "NA", "NA"]
    df["Dice2"] = [dice_s2_siren, "NA", "NA"]

    df["IoU1"] = [iou_s1_siren, "NA", "NA"]
    df["IoU2"] = [iou_s2_siren, "NA", "NA"]

    # print(df)
    return df
