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
import monai
# from Models.models import Siren, Finer
import Models.models as models
import monai
import pandas as pd

#Helper Functions
norm = lambda img : (img - img.min())/(img.max() - img.min())
dice_stack_helper = lambda img : torch.stack((img[:,:,3].reshape(size), img[:,:,0].reshape(size), img[:,:,1].reshape(size), img[:,:,2].reshape(size)), dim=0).unsqueeze(0) #Expected shape = (1,*hf_spatial,4)

def get_model(config):
    if(config["model"] == 0):
        return models.Siren(in_features=config["in_features"], out_features=8, hidden_features=config["hidden_features"], hidden_layers=config["hidden_layers"], outermost_linear=True, first_omega_0 = config["w0"], hidden_omega_0 = config["w0"])
    else:
        return models.Finer(in_features=config["in_features"], out_features=8, hidden_layers=config["hidden_layers"], hidden_features=config["hidden_features"], first_omega=config["w0"], hidden_omega=config["w0"], init_method='sine', init_gain=1, fbs=None)






def get_device():
    
    if(torch.cuda.is_available()):
        device = torch.device("cuda")[0]
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
    norm = lambda img : (img - img.min())/(img.max() - img.min())

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

    '''
    print("Dice Score between SIREN Pred and GT: ", dice_score(hf_gt_seg1, dice_stack_helper(model_output_seg1)).mean().item()) #l2 = 1e-1 gave 0.3288/0.3291, 1e-2 gave 0.3361/0.3301, l2 = 1 gave 0.3309
    print("IoU Score between SIREN Pred and GT: ", iou_score(hf_gt_seg1, dice_stack_helper(model_output_seg1)).mean().item())


    print("Bilinear : ", interpolation_metrics(interpolated_bilinear[0], hf_ground_truth1))
    print("Bicubic : ", interpolation_metrics(interpolated_bicubic[0], hf_ground_truth1))
    print("SIREN output : ", interpolation_metrics(get_full_img(model_output_seg1, model_output_img1, config).unsqueeze(0).unsqueeze(0).to('cpu'), hf_ground_truth1))


    print("-----------")
    print("Subject = 2")




    # model_output_seg2 = model_output_seg_list2[0]
    # hf_gt_seg_2 = hf_gt_seg2[0]#.to(get_device())


    print("Dice Score between SIREN Pred and GT: ", dice_score(hf_gt_seg2, dice_stack_helper(model_output_seg2)).mean().item()) #l2 = 1e-1 gave 0.3288/0.3291, 1e-2 gave 0.3361/0.3301, l2 = 1 gave 0.3309
    print("IoU Score between SIREN Pred and GT: ", iou_score(hf_gt_seg2, dice_stack_helper(model_output_seg2)).mean().item())


    print("Bilinear : ", interpolation_metrics(interpolated_bilinear[1], hf_ground_truth2))
    print("Bicubic : ", interpolation_metrics(interpolated_bicubic[1], hf_ground_truth2))
    print("SIREN output : ", interpolation_metrics(get_full_img(model_output_seg2, model_output_img2, config).unsqueeze(0).unsqueeze(0).to('cpu'), hf_ground_truth2))
    # print("SIREN+FFE output : ", interpolation_metrics(get_full_img(model_output_seg_f2, model_output_img_f2, config).unsqueeze(0).unsqueeze(0).to('cpu'), hf_ground_truth_2))

    '''
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

    '''
    print("Dice Score between SIREN Pred and GT: ", dice_score(hf_gt_seg1, dice_stack_helper(model_output_seg1)).mean().item()) #l2 = 1e-1 gave 0.3288/0.3291, 1e-2 gave 0.3361/0.3301, l2 = 1 gave 0.3309
    print("IoU Score between SIREN Pred and GT: ", iou_score(hf_gt_seg1, dice_stack_helper(model_output_seg1)).mean().item())


    print("Bilinear : ", interpolation_metrics(interpolated_bilinear[0], hf_ground_truth1))
    print("Bicubic : ", interpolation_metrics(interpolated_bicubic[0], hf_ground_truth1))
    print("SIREN output : ", interpolation_metrics(get_full_img(model_output_seg1, model_output_img1, config).unsqueeze(0).unsqueeze(0).to('cpu'), hf_ground_truth1))


    print("-----------")
    print("Subject = 2")




    # model_output_seg2 = model_output_seg_list2[0]
    # hf_gt_seg_2 = hf_gt_seg2[0]#.to(get_device())


    print("Dice Score between SIREN Pred and GT: ", dice_score(hf_gt_seg2, dice_stack_helper(model_output_seg2)).mean().item()) #l2 = 1e-1 gave 0.3288/0.3291, 1e-2 gave 0.3361/0.3301, l2 = 1 gave 0.3309
    print("IoU Score between SIREN Pred and GT: ", iou_score(hf_gt_seg2, dice_stack_helper(model_output_seg2)).mean().item())


    print("Bilinear : ", interpolation_metrics(interpolated_bilinear[1], hf_ground_truth2))
    print("Bicubic : ", interpolation_metrics(interpolated_bicubic[1], hf_ground_truth2))
    print("SIREN output : ", interpolation_metrics(get_full_img(model_output_seg2, model_output_img2, config).unsqueeze(0).unsqueeze(0).to('cpu'), hf_ground_truth2))
    # print("SIREN+FFE output : ", interpolation_metrics(get_full_img(model_output_seg_f2, model_output_img_f2, config).unsqueeze(0).unsqueeze(0).to('cpu'), hf_ground_truth_2))

    '''
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
