import torchio as tio
import torch
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
from Utils.utils import dice_stack_helper
from Utils.defaults import default_config
from LFSynth.ContrastEstimation import forward as contrast_forward
from LFSynth.ContrastEstimation import read_imgs, get_hf_tissue_seg

import random
from Utils.utils import get_device, norm
import sys
from PIL import Image
from Data.ImagePreparation import ImagePreparation



# To download full IXI dataset use torchio - https://docs.torchio.org/datasets.html#module-torchio.datasets.ixi






def get_lf_observed_segmentations(device, dataset_num):
    lf_wm_location = "./Data/ixi/T1/" + str(dataset_num) + "/ulf/fast_pve_2.nii.gz"
    lf_gm_location = "./Data/ixi/T1/" + str(dataset_num) + "/ulf/fast_pve_1.nii.gz"
    lf_csf_location = "./Data/ixi/T1/" + str(dataset_num) + "/ulf/fast_pve_0.nii.gz"

    lf_wm_seg = torch.from_numpy(nib.load(lf_wm_location).get_fdata())
    lf_gm_seg = torch.from_numpy(nib.load(lf_gm_location).get_fdata())
    lf_csf_seg = torch.from_numpy(nib.load(lf_csf_location).get_fdata())
    # print(lf_wm_seg.shape)
    total_lf_seg = lf_wm_seg + lf_gm_seg + lf_csf_seg
    lf_bg_seg = 1 - total_lf_seg
    return lf_wm_seg.flatten().to(torch.float32).to(device).unsqueeze(0), lf_gm_seg.flatten().to(torch.float32).to(device).unsqueeze(0), lf_csf_seg.flatten().to(torch.float32).to(device).unsqueeze(0), lf_bg_seg.flatten().to(torch.float32).to(device).unsqueeze(0)






def get_hf_observed_segmentations(dataset_num, config):
    folder = './Data/ixi/T1/' + str(dataset_num) + "/"
    slice = config["slice"]
    (img_nib, wm_nib, gm_nib, csf_nib) = read_imgs(folder)
    (wm_obs_seg, gm_obs_seg, csf_obs_seg, bg_obs_seg) = get_hf_tissue_seg(wm_nib, gm_nib, csf_nib)
    (wm_obs_seg, gm_obs_seg, csf_obs_seg, bg_obs_seg) = (torch.from_numpy(wm_obs_seg), torch.from_numpy(gm_obs_seg), torch.from_numpy(csf_obs_seg), torch.from_numpy(bg_obs_seg))
    # (wm_gt_seg, gm_gt_seg, csf_gt_seg, bg_gt_seg) = (wm_gt_seg[:,:,slice], gm_gt_seg[:,:,slice], csf_gt_seg[:,:,slice], bg_gt_seg[:,:,slice])
    return (wm_obs_seg, gm_obs_seg, csf_obs_seg, bg_obs_seg)





def load_data(dataset_num, config=default_config):
    slice = default_config["slice"]
    # random.seed(9600) #For Reproducibility -> using pl.seed_everything in main()
    device = get_device() #Returns either MPS/CUDA/CPU depending on availability

    lf_wm_seg, lf_gm_seg, lf_csf_seg, lf_bg_seg = get_lf_observed_segmentations(device, dataset_num) #Load ULF Observed Segmentations
    
    #Load HF observed
    hf_loc = "./Data/ixi/T1/" + str(dataset_num)+ "/fast_restore.nii.gz"
    hf_observed = nib.load(hf_loc).get_fdata()
    
    (wm_lf_like, gm_lf_like, csf_lf_like, bg_lf_like, lf_like), (wm_seg, gm_seg, csf_seg, bg_seg), (wm_snr, gm_snr, csf_snr), M = contrast_forward(dataset_num) #Generating ULF observed
    print(wm_lf_like.shape, lf_like.shape, wm_seg.shape, wm_snr, M)
    #Load ULF observed
    lf_observed = norm(lf_like) #normalized
    
    
    # orig_img = Image.fromarray(lf_observed)
    
    #Updating size config parameters
    config["size"] = (hf_observed.shape[0], hf_observed.shape[1], hf_observed.shape[2]) #Loading 2D Images. For datasets = 1, 2 (174, 192)
    config["size_lf"] = (lf_observed.shape[0], lf_observed.shape[1], lf_observed.shape[2])
    # config["M"] = M
    

    
    lf_wm_seg, lf_gm_seg, lf_csf_seg, lf_bg_seg = get_lf_observed_segmentations(device, dataset_num) #Load ULF Observed Segmentations
    lf_observed_seg_dice = torch.stack((lf_bg_seg[0].reshape(config["size_lf"]), lf_wm_seg[0].reshape(config["size_lf"]), lf_gm_seg[0].reshape(config["size_lf"]), lf_csf_seg[0].reshape(config["size_lf"])), dim=0).unsqueeze(0)
 

    # img_prep_obj = ImagePreparation(lf_observed_seg_dice, config["size_lf"][0], config["size_lf"][1], is_ffe=config["ffe"]) #Image Preparation Object
    # lf_dataloader = DataLoader(img_prep_obj, batch_size=1, pin_memory=True, num_workers=0)
    # config["in_features"] = 256 if(config["ffe"]==True) else  2
    print(lf_wm_seg.shape, lf_observed_seg_dice.shape)

    # return hf_observed, lf_dataloader, lf_observed, M #might need to uncomment this. lf_dataloader is the default used everywhere
    return hf_observed, lf_observed, lf_observed_seg_dice, M #might need to comment this




