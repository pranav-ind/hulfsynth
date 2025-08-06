import torch
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
from Utils.utils import dice_stack_helper
from Utils.defaults import default_config
from LFSynth.ContrastEstimation import forward as contrast_forward
from LFSynth.ContrastEstimation import read_imgs, tissue_probabilities
import random
from Utils.utils import get_device, norm
import sys
from PIL import Image
from Data.ImagePreparation import ImagePreparation


def priors(device, dataset_num):
    '''
    Input : Device, Dataset_num = {1, 2}
    *spatial = H * W * slices
    Given that the probabilty maps at each voxel ≠ 1 at every voxel after registration, applying a softmax function
    Returns (WM, GM, CSF, BG) each of shape( 1, *spatial)
    '''
    wm_location = "./Priors/icbm152_2009/mni_icbm152_nlin_sym_09a/Registered_dataset_" + str(dataset_num) + "/reg/wm_reg.nii.gz"
    gm_location = "./Priors/icbm152_2009/mni_icbm152_nlin_sym_09a/Registered_dataset_" + str(dataset_num) + "/reg/gm_reg.nii.gz"
    csf_location = "./Priors/icbm152_2009/mni_icbm152_nlin_sym_09a/Registered_dataset_" + str(dataset_num) + "/reg/csf_reg.nii.gz"
    
    wm_prior = torch.from_numpy(nib.load(wm_location).get_fdata())
    gm_prior = torch.from_numpy(nib.load(gm_location).get_fdata())
    csf_prior = torch.from_numpy(nib.load(csf_location).get_fdata())
    total_seg = wm_prior + gm_prior + csf_prior #Raw total segmentations which is > 1 in many voxels
    
    #Creating BG tissue
    bg_mask = torch.where(total_seg>0, 1, 0) #Binary masking all the foreground voxels to ensure that only bg pixels are considered
    bg_prior = 1 - bg_mask 
    total_seg = wm_prior + gm_prior + csf_prior + bg_prior 
    stacked_seg = torch.stack((wm_prior, gm_prior, csf_prior, bg_prior), dim=-1)
    stacked_seg = torch.nn.Softmax(dim=-1)(stacked_seg)
    
    wm_prior, gm_prior, csf_prior, bg_prior = stacked_seg[:,:,:,0], stacked_seg[:,:,:,1], stacked_seg[:,:,:,2], stacked_seg[:,:,:,3]
    return wm_prior.to(torch.float32).to(device).flatten().unsqueeze(0), gm_prior.to(torch.float32).to(device).flatten().unsqueeze(0), csf_prior.to(torch.float32).to(device).flatten().unsqueeze(0), bg_prior.to(torch.float32).to(device).flatten().unsqueeze(0) #For Model Training
    


def lf_segmentations(device, dataset_num):
    lf_wm_location = "./Data/data_" + str(dataset_num) + "/lf_seg/lf_seg_pve_2.nii.gz"
    lf_gm_location = "./Data/data_" + str(dataset_num) + "/lf_seg/lf_seg_pve_1.nii.gz"
    lf_csf_location = "./Data/data_" + str(dataset_num) + "/lf_seg/lf_seg_pve_0.nii.gz"

    lf_wm_seg = torch.from_numpy(nib.load(lf_wm_location).get_fdata())
    lf_gm_seg = torch.from_numpy(nib.load(lf_gm_location).get_fdata())
    lf_csf_seg = torch.from_numpy(nib.load(lf_csf_location).get_fdata())
    print(lf_wm_seg.shape)
    total_lf_seg = lf_wm_seg + lf_gm_seg + lf_csf_seg
    lf_bg_seg = 1 - total_lf_seg
    return lf_wm_seg.flatten().to(torch.float32).to(device).unsqueeze(0), lf_gm_seg.flatten().to(torch.float32).to(device).unsqueeze(0), lf_csf_seg.flatten().to(torch.float32).to(device).unsqueeze(0), lf_bg_seg.flatten().to(torch.float32).to(device).unsqueeze(0)






def get_gt_seg(dataset_num, config):
    folder = "./Data/data_" + str(dataset_num) + "/"
    slice = config["slice"]
    (img_nib, wm_nib, gm_nib, csf_nib) = read_imgs(folder)
    (wm_gt_seg, gm_gt_seg, csf_gt_seg, bg_gt_seg) = tissue_probabilities(wm_nib, gm_nib, csf_nib)
    (wm_gt_seg, gm_gt_seg, csf_gt_seg, bg_gt_seg) = (torch.from_numpy(wm_gt_seg), torch.from_numpy(gm_gt_seg), torch.from_numpy(csf_gt_seg), torch.from_numpy(bg_gt_seg))
    # (wm_gt_seg, gm_gt_seg, csf_gt_seg, bg_gt_seg) = (wm_gt_seg[:,:,slice], gm_gt_seg[:,:,slice], csf_gt_seg[:,:,slice], bg_gt_seg[:,:,slice])
    return (wm_gt_seg, gm_gt_seg, csf_gt_seg, bg_gt_seg)





def load_data(dataset_num, config=default_config):
    slice = default_config["slice"]
    random.seed(9600) #For Reproducibility
    device = get_device() #Returns either MPS/CUDA/CPU depending on availability

    lf_wm_seg, lf_gm_seg, lf_csf_seg, lf_bg_seg = lf_segmentations(device, dataset_num) #Load ULF GT Segmentations
    
    #Load HF GT
    hf_loc = "./Data/data_" + str(dataset_num)+ "/fast_restore.nii.gz"
    hf_ground_truth = nib.load(hf_loc).get_fdata()
    
    (wm_lf_like, gm_lf_like, csf_lf_like, bg_lf_like, lf_like), (wm_prob, gm_prob, csf_prob, bg_prob), (wm_snr, gm_snr, csf_snr), M = contrast_forward(dataset_num) #Generating ULF GT
    print(wm_lf_like.shape, lf_like.shape, wm_prob.shape, wm_snr, M)
    #Load ULF GT
    lf_gt = lf_like
    lf_gt = norm(lf_gt)
    
    # orig_img = Image.fromarray(lf_gt)
    
    #Updating size config parameters
    config["size"] = (hf_ground_truth.shape[0], hf_ground_truth.shape[1], hf_ground_truth.shape[2]) #Loading 2D Images. For datasets = 1, 2 (174, 192)
    config["size_lf"] = (lf_gt.shape[0], lf_gt.shape[1], lf_gt.shape[2])
    # config["M"] = M
    
    wm_prior_seg, gm_prior_seg, csf_prior_seg, bg_prior_seg = priors(device, dataset_num) #Load Priors
    prior_seg_dice = torch.stack((bg_prior_seg[0].reshape(config["size"]), wm_prior_seg[0].reshape(config["size"]), gm_prior_seg[0].reshape(config["size"]), csf_prior_seg[0].reshape(config["size"])), dim=0).unsqueeze(0) 
    
    lf_wm_seg, lf_gm_seg, lf_csf_seg, lf_bg_seg = lf_segmentations(device, dataset_num) #Load ULF GT Segmentations
    lf_gt_seg_dice = torch.stack((lf_bg_seg[0].reshape(config["size_lf"]), lf_wm_seg[0].reshape(config["size_lf"]), lf_gm_seg[0].reshape(config["size_lf"]), lf_csf_seg[0].reshape(config["size_lf"])), dim=0).unsqueeze(0)
 

    img_prep_obj = ImagePreparation(lf_gt, config["size_lf"][0], config["size_lf"][1], is_ffe=config["ffe"]) #Image Preparation Object
    lf_dataloader = DataLoader(img_prep_obj, batch_size=1, pin_memory=True, num_workers=0)
    # config["in_features"] = 256 if(config["ffe"]==True) else  2
    print(wm_prior_seg.shape, prior_seg_dice.shape, lf_wm_seg.shape, lf_gt_seg_dice.shape, )

    # return hf_ground_truth, lf_dataloader, prior_seg_dice, lf_gt_seg_dice, M #might need to uncomment this. lf_dataloader is the default used everywhere
    return hf_ground_truth, lf_gt, prior_seg_dice, lf_gt_seg_dice, M #might need to comment this
