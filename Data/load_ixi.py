import torchio as tio
import torch
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
from Utils.utils import dice_stack_helper, segment
from Utils.defaults import default_config
from LFSynth.ContrastEstimation import forward as contrast_forward
from LFSynth.ContrastEstimation import read_imgs, get_hf_tissue_seg

import random
from Utils.utils import get_device, norm
import sys
from PIL import Image
from Data.ImagePreparation import ImagePreparation
import numpy as np
import ast


# To download full IXI dataset use torchio - https://docs.torchio.org/datasets.html#module-torchio.datasets.ixi






def get_lf_observed_segmentations(dataset_num, config):
    #TODO: parse config = default_config
    if(config["is_new_contrast"]):
        print('segmenting the image... (Flag config["is_new_contrast"]) ')
        path = './Data/ixi/T1/' + str(dataset_num) + '/ulf/'
        img_loc = './Data/ixi/T1/' + str(dataset_num) + '/ulf/ulf_temp.nii.gz'
        segment(img_loc, path)
    
    lf_wm_location = "./Data/ixi/T1/" + str(dataset_num) + "/ulf/fast_pve_2.nii.gz"
    lf_gm_location = "./Data/ixi/T1/" + str(dataset_num) + "/ulf/fast_pve_1.nii.gz"
    lf_csf_location = "./Data/ixi/T1/" + str(dataset_num) + "/ulf/fast_pve_0.nii.gz"

    lf_wm_seg = torch.from_numpy(nib.load(lf_wm_location).get_fdata())
    lf_gm_seg = torch.from_numpy(nib.load(lf_gm_location).get_fdata())
    lf_csf_seg = torch.from_numpy(nib.load(lf_csf_location).get_fdata())
    # print(lf_wm_seg.shape)
    total_lf_seg = lf_wm_seg + lf_gm_seg + lf_csf_seg
    lf_bg_seg = 1 - total_lf_seg
    # return lf_wm_seg.flatten().to(torch.float32).to(device).unsqueeze(0), lf_gm_seg.flatten().to(torch.float32).to(device).unsqueeze(0), lf_csf_seg.flatten().to(torch.float32).to(device).unsqueeze(0), lf_bg_seg.flatten().to(torch.float32).to(device).unsqueeze(0)
    return lf_wm_seg.flatten().to(torch.float32).unsqueeze(0), lf_gm_seg.flatten().to(torch.float32).unsqueeze(0), lf_csf_seg.flatten().to(torch.float32).unsqueeze(0), lf_bg_seg.flatten().to(torch.float32).unsqueeze(0)



def get_lf_observed_segmentations_sens(sens_folder, config):
    
    lf_wm_location = sens_folder + "/fast_pve_2.nii.gz"
    lf_gm_location = sens_folder + "/fast_pve_1.nii.gz"
    lf_csf_location = sens_folder + "/fast_pve_0.nii.gz"

    lf_wm_seg = torch.from_numpy(nib.load(lf_wm_location).get_fdata())
    lf_gm_seg = torch.from_numpy(nib.load(lf_gm_location).get_fdata())
    lf_csf_seg = torch.from_numpy(nib.load(lf_csf_location).get_fdata())
    # print(lf_wm_seg.shape)
    total_lf_seg = lf_wm_seg + lf_gm_seg + lf_csf_seg
    lf_bg_seg = 1 - total_lf_seg
    # return lf_wm_seg.flatten().to(torch.float32).to(device).unsqueeze(0), lf_gm_seg.flatten().to(torch.float32).to(device).unsqueeze(0), lf_csf_seg.flatten().to(torch.float32).to(device).unsqueeze(0), lf_bg_seg.flatten().to(torch.float32).to(device).unsqueeze(0)
    return lf_wm_seg.flatten().to(torch.float32).unsqueeze(0), lf_gm_seg.flatten().to(torch.float32).unsqueeze(0), lf_csf_seg.flatten().to(torch.float32).unsqueeze(0), lf_bg_seg.flatten().to(torch.float32).unsqueeze(0)




def get_hf_observed_segmentations(dataset_num, config):
    dataset_num = config["dataset_num"]
    folder = './Data/ixi/T1/' + str(dataset_num) + '/'
    slice = config["slice"]
    (img_nib, wm_nib, gm_nib, csf_nib) = read_imgs(folder)
    (wm_obs_seg, gm_obs_seg, csf_obs_seg, bg_obs_seg) = get_hf_tissue_seg(wm_nib, gm_nib, csf_nib)
    (wm_obs_seg, gm_obs_seg, csf_obs_seg, bg_obs_seg) = (torch.from_numpy(wm_obs_seg), torch.from_numpy(gm_obs_seg), torch.from_numpy(csf_obs_seg), torch.from_numpy(bg_obs_seg))
    # (wm_gt_seg, gm_gt_seg, csf_gt_seg, bg_gt_seg) = (wm_gt_seg[:,:,slice], gm_gt_seg[:,:,slice], csf_gt_seg[:,:,slice], bg_gt_seg[:,:,slice])
    return (wm_obs_seg, gm_obs_seg, csf_obs_seg, bg_obs_seg)





def load_data(config=default_config):
    slice = config["slice"]
    dataset_num = config["dataset_num"]
    # random.seed(9600) #For Reproducibility -> using pl.seed_everything in main()
    # device = get_device() #Returns either MPS/CUDA/CPU depending on availability

    # lf_wm_seg, lf_gm_seg, lf_csf_seg, lf_bg_seg = get_lf_observed_segmentations(dataset_num) #Load ULF Observed Segmentations
    
    #Load HF observed
    hf_loc = "./Data/ixi/T1/" + str(dataset_num)+ "/hf/fast_restore.nii.gz"
    hf_observed = nib.load(hf_loc).get_fdata()
    
    (wm_lf_like, gm_lf_like, csf_lf_like, bg_lf_like, lf_like), (wm_seg, gm_seg, csf_seg, bg_seg), M = contrast_forward(dataset_num) #Generating ULF observed
    config["M"] = M
    # print(wm_lf_like.shape, lf_like.shape, wm_seg.shape, M)
    #Load ULF observed
    lf_observed = norm(lf_like) #normalized
    
    
    #Updating size config parameters
    config["size"] = (hf_observed.shape[0], hf_observed.shape[1], hf_observed.shape[2]) #Loading 2D Images. For datasets = 1, 2 (174, 192)
    config["size_lf"] = (lf_observed.shape[0], lf_observed.shape[1], lf_observed.shape[2])
    
 
    lf_wm_seg, lf_gm_seg, lf_csf_seg, lf_bg_seg = get_lf_observed_segmentations(dataset_num, config) #Load ULF Observed Segmentations
    lf_observed_seg_dice = torch.stack((lf_bg_seg[0].reshape(config["size_lf"]), lf_wm_seg[0].reshape(config["size_lf"]), lf_gm_seg[0].reshape(config["size_lf"]), lf_csf_seg[0].reshape(config["size_lf"])), dim=0).unsqueeze(0)
 

    # return hf_observed, lf_dataloader, lf_observed, M #might need to uncomment this. lf_dataloader is the default used everywhere
    return hf_observed, lf_observed, lf_observed_seg_dice, M #might need to comment this



def load_sensitivity_data(config=default_config):
    slice = config["slice"]
    dataset_num = config["dataset_num"]
    sens_id = config["sens_id"]
    folder = './Data/ixi/T1/' + str(dataset_num) + "/"
    sens_folder = folder + 'sensitivity_data/contrast/' + str(sens_id)

    #Load HF observed
    hf_loc = folder + "/hf/fast_restore.nii.gz"
    hf_observed = nib.load(hf_loc).get_fdata()
    
    # (wm_lf_like, gm_lf_like, csf_lf_like, bg_lf_like, lf_like), (wm_seg, gm_seg, csf_seg, bg_seg), M = contrast_forward(dataset_num) #Generating ULF observed
    lf_loc = sens_folder + '/brain.nii.gz'
    lf_like = nib.load(lf_loc).get_fdata()
    lf_observed = norm(lf_like) #normalized
    lines = []
    try:
        with open( sens_folder + '/cnrs.txt', 'r') as file:
            for line in file:
                # print(ast.literal_eval(line), type(ast.literal_eval(line)))
                # temp = ast.literal_eval(((line.strip()).split(' ', 1)[1]))
                lines.append(ast.literal_eval(line))
                
                
    except FileNotFoundError:
        print(f"Error: The file '{sens_folder + '/cnrs.txt'}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    M = lines[-1]
    print(M)
    config["M"] = M

    #Updating size config parameters
    config["size"] = (hf_observed.shape[0], hf_observed.shape[1], hf_observed.shape[2]) #Loading 2D Images. For datasets = 1, 2 (174, 192)
    config["size_lf"] = (lf_observed.shape[0], lf_observed.shape[1], lf_observed.shape[2])
    

    lf_wm_seg, lf_gm_seg, lf_csf_seg, lf_bg_seg = get_lf_observed_segmentations_sens(sens_folder, config)
    # lf_wm_seg, lf_gm_seg, lf_csf_seg, lf_bg_seg = get_lf_observed_segmentations(dataset_num, config) #Load ULF Observed Segmentations
    lf_observed_seg_dice = torch.stack((lf_bg_seg[0].reshape(config["size_lf"]), lf_wm_seg[0].reshape(config["size_lf"]), lf_gm_seg[0].reshape(config["size_lf"]), lf_csf_seg[0].reshape(config["size_lf"])), dim=0).unsqueeze(0)


    # return hf_observed, lf_dataloader, lf_observed, M #might need to uncomment this. lf_dataloader is the default used everywhere
    return hf_observed, lf_observed, lf_observed_seg_dice, M #might need to comment this

