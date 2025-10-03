import torch
import pandas as pd
import numpy as np
import torchvision.transforms as Tr
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import nibabel as nib
import matplotlib as mpl
from skimage.exposure import match_histograms
# from nilearn import image
# from nilearn import plotting
# from tkinter import filedialog as fd
from scipy.ndimage import gaussian_filter
import math
from scipy import optimize, stats
from scipy.optimize import NonlinearConstraint, LinearConstraint, minimize, Bounds, least_squares
import itertools
from sklearn.metrics import mean_squared_error
import matplotlib.patches as patches
import pandas as pd 
from fsl.wrappers.fast import fast as fast
import time
from Utils.defaults import default_config
from Utils.utils import norm
import ast

def read_ulf_imgs(folder):
    #Returns observed image and segmentations in nib format
    folder = folder + "ulf/fast"
    csf_nib = nib.load(folder + "_pve_0.nii.gz") 
    gm_nib = nib.load(folder + "_pve_1.nii.gz") 
    wm_nib = nib.load(folder + "_pve_2.nii.gz") 
    img_nib = nib.load(folder + "_restore.nii.gz")
    return (img_nib, wm_nib, gm_nib, csf_nib)

def get_hf_observed(folder):
    #Returns observed image and segmentations in nib format
    folder = folder + "hf/fast"
    hf_csf = nib.load(folder + "_pve_0.nii.gz") 
    hf_gm = nib.load(folder + "_pve_1.nii.gz") 
    hf_wm = nib.load(folder + "_pve_2.nii.gz") 
    hf_img = nib.load(folder + "_restore.nii.gz")
    return (hf_img, hf_wm, hf_gm, hf_csf)


def get_tissue_seg(wm_nib, gm_nib, csf_nib):
    '''
    Adds up all probability segmentations and creates a new segmentation for background tissue. Expected : sum of all (4) tissue probabilities = 1
    Returns a tuple of probabilities
    '''
    csf =  csf_nib.get_fdata()  
    gm =   gm_nib.get_fdata() 
    wm =   wm_nib.get_fdata()
    total_seg = csf + gm + wm
    total_seg = total_seg.clip(max=1) #Because for some voxels the summation is giving value slightly greater than 1.0 (ex : 1.0000000298023224). So clipping max value to 1
    bg = 1 - total_seg
    bg = np.floor(bg)
    return (wm, gm, csf, bg)


def seg_to_intenities(img_nib, wm_nib, gm_nib, csf_nib, bg):
    '''
    Expecting Tissue Probability Segmentations of WM, GM, CSF, BG and Converting them to Image Intensities.
    Note : dtype of BG tissue probability map = numpy array while all others are NII images
    '''
    csf =  img_nib.get_fdata() * csf_nib.get_fdata()  #* csf_masked  
    gm =   img_nib.get_fdata() * gm_nib.get_fdata() #* gm_masked
    wm =   img_nib.get_fdata() * wm_nib.get_fdata() #* wm_masked
    bg =   img_nib.get_fdata() * bg
    hf_img = csf + gm + wm + bg

    return (wm, gm, csf, bg, hf_img)

def calc_val_ulf_snr(img_nib, wm_nib, gm_nib, csf_nib, dataset):
    rayleigh_correction = 1.53
    if(dataset == 'val'):
        #ROIs are extracted manually. For example, refer roi_gen.ipynb 
        slice_num = 81
        wm_roi_pixels = img_nib.get_fdata()[86:94, 157:167, slice_num][wm_nib.get_fdata()[86:94, 157:167, slice_num]>0.9]
        gm_roi_pixels = img_nib.get_fdata()[123:127, 132:144, slice_num][gm_nib.get_fdata()[123:127, 132:144, slice_num]>0.9]
        csf_roi_pixels = img_nib.get_fdata()[104:114, 125:133, slice_num][csf_nib.get_fdata()[104:114, 125:133, slice_num]>0.9]
    
    noise_file = "./Data/validation_data/sub_" + dataset + "ulf/raw.nii.gz"
    noise = nib.load(noise_file)
    # print("BG Noise in different regions :", noise.get_fdata()[:20,:20,95].std(), noise.get_fdata()[150:175,150:175,95].std(), noise.get_fdata()[155:,:25,95].std(), noise.get_fdata()[:25,150:170,95].std()) #Mean of all 4 regions in background
    
    std_bg = noise.get_fdata()[:,:,slice_num][168:182, 20:28, ].std() #Background Noise extracted manually

    csf_snr = csf_roi_pixels.mean()/ (std_bg * rayleigh_correction)
    gm_snr = gm_roi_pixels.mean()/ (std_bg * rayleigh_correction)
    wm_snr = wm_roi_pixels.mean()/ (std_bg * rayleigh_correction)
    print("ULF SNRs for WM: ", wm_snr, "GM: ", gm_snr, "CSF: ", csf_snr, "BG (std): ", std_bg)

    Ccg = abs(csf_snr - gm_snr) 
    Ccw = abs(csf_snr - wm_snr)
    Cgw = abs(gm_snr - wm_snr)

    print("CNRs- WG: ", Cgw, "WC: ", Ccw, "GC: ", Ccg)
    return (wm_snr, gm_snr, csf_snr)

def get_rois(dataset_num='0011', field_type='hf'):
    file_path = './Data/validation_data/sub_' + dataset_num + '/' + field_type + '/snrs.txt'
    lines = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                lines.append(ast.literal_eval(((line.strip()).split(' ', 1)[1])))
                
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    roi_voxels = lines[:4]
    roi_snrs = lines[4:]
    return roi_voxels, roi_snrs

'''
def get_target_c(img_nib, wm_nib, gm_nib, csf_nib, dataset_num, field_type='hf'):
    #HF input -> HF output
    rayleigh_correction = 1.53
    if(dataset == '0011'):
        #ROIs are extracted manually. For example, refer roi_gen.ipynb 
        slice_num = 11
        roi_voxel_list, _ = get_rois(dataset, field_type)
        wm_roi_pixels = img_nib.get_fdata()[roi_voxel_list[0][1]:roi_voxel_list[0][0]+roi_voxel_list[0][3], roi_voxel_list[0][0]:roi_voxel_list[0][0]+roi_voxel_list[0][2], slice_num][wm_nib.get_fdata()[roi_voxel_list[0][1]:roi_voxel_list[0][0]+roi_voxel_list[0][3], roi_voxel_list[0][0]:roi_voxel_list[0][0]+roi_voxel_list[0][2], slice_num]>0.9]
        gm_roi_pixels = img_nib.get_fdata()[roi_voxel_list[1][1]:roi_voxel_list[1][0]+roi_voxel_list[1][3], roi_voxel_list[1][0]:roi_voxel_list[1][0]+roi_voxel_list[1][2], slice_num][gm_nib.get_fdata()[roi_voxel_list[1][1]:roi_voxel_list[1][0]+roi_voxel_list[1][3], roi_voxel_list[1][0]:roi_voxel_list[1][0]+roi_voxel_list[1][2], slice_num]>0.9]
        csf_roi_pixels = img_nib.get_fdata()[roi_voxel_list[2][1]:roi_voxel_list[2][0]+roi_voxel_list[2][3], roi_voxel_list[2][0]:roi_voxel_list[2][0]+roi_voxel_list[2][2], slice_num][csf_nib.get_fdata()[roi_voxel_list[2][1]:roi_voxel_list[2][0]+roi_voxel_list[2][3], roi_voxel_list[2][0]:roi_voxel_list[2][0]+roi_voxel_list[2][2], slice_num]>0.9]
    
    noise_file = './Data/validation_data/sub_' + dataset_num + '/' + field_type + '/raw.nii.gz'
    noise = nib.load(noise_file)
    # print("BG Noise in different regions :", noise.get_fdata()[:20,:20,95].std(), noise.get_fdata()[150:175,150:175,95].std(), noise.get_fdata()[155:,:25,95].std(), noise.get_fdata()[:25,150:170,95].std()) #Mean of all 4 regions in background
    
    std_bg = noise.get_fdata()[:,:,slice_num][roi_voxel_list[3][1]:roi_voxel_list[3][0]+roi_voxel_list[3][3], roi_voxel_list[3][0]:roi_voxel_list[3][0]+roi_voxel_list[3][2]].std() #Background Noise extracted manually

    csf_snr = csf_roi_pixels.mean()/ (std_bg * rayleigh_correction)
    gm_snr = gm_roi_pixels.mean()/ (std_bg * rayleigh_correction)
    wm_snr = wm_roi_pixels.mean()/ (std_bg * rayleigh_correction)
    print(field_type + "SNRs for WM: ", wm_snr, "GM: ", gm_snr, "CSF: ", csf_snr, "BG (std): ", std_bg)

    Ccg = abs(csf_snr - gm_snr) 
    Ccw = abs(csf_snr - wm_snr)
    Cgw = abs(gm_snr - wm_snr)
    target_c = np.array([Cgw, Ccw, Ccg])
    print("CNRs- WG: ", Cgw, "WC: ", Ccw, "GC: ", Ccg, "target_c: ",target_c)
    return  target_c #Returns target_c
'''
def get_target_c(dataset_num='0011', target_type='hf'):
    _, roi_snrs = get_rois(dataset_num='0011', field_type=target_type) #returns HF SNRs if target_type == HF
    S_list = roi_snrs
    target_c = np.array([S_list[0]-S_list[1], S_list[0] -S_list[2], S_list[1] -S_list[2]]) #target_c = [Cwg, Cwc, Cgc]
    return target_c


def get_m(dataset_num='0011', target_type='hf'):
    #m^{ULF} = get_m (c^{HF}, S^{ULF}) i.e., if target_type == ULF then S matrix should be from HF and if target_type == HF then S matrix should be ULF ;
    if(target_type == 'ulf'):
        S_type = 'hf'
    else:
        S_type = 'ulf'
    rayleigh_correction = 1.53
    _, S_list = get_rois(dataset_num, field_type=S_type)
    s = np.array([[S_list[0], -S_list[1], 0], [S_list[0], 0, -S_list[2]], [0, S_list[1], -S_list[2]]]) #SNR matrix #S_list[0] = WM, S_list[1] = GM, S_list[2] = CSF
    target_c = get_target_c(dataset_num, target_type)
    print("s, c: ", s.shape, target_c.shape)
    grid = GridSearch(s,target_c, upper_bound=np.inf)
    grid_results = grid.solve()
    grid_results = grid.easy_results(grid_results)
    print(grid_results)
    M = grid_results.iloc[0].x
    return s, target_c, M


    



def toy_values(wm_snr, gm_snr, csf_snr, dataset):
    # known_m = np.array([0.6, 0.6, 0.3]) #other values
    # known_m = np.array([0.7, 0.3, 0.2]) #other values
    # known_m = np.array([0.5, 0.6, 0.2]) #dataset = 1 #base

    if(dataset ==1):
        # known_m = np.array([0.75, 0.85, 0.9]) #dataset = 1 #id = 1
        known_m = np.array([0.5, 0.6, 0.2]) #dataset = 1 #base
        # known_m = np.array([0.4, 0.5, 0.5]) #dataset = 1 #id = 2
        # known_m = np.array([0.6, 0.6, 0.3]) #dataset = 1 #id = 3
        # known_m = np.array([0.5, 0.5, 0.2]) #dataset = 1 #id = 4
        known_m = np.array([0.75, 0.9, 0.9])


    else:
        # known_m = np.array([0.4, 0.5, 0.2]) #dataset = 2
        known_m = np.array([0.5, 0.6, 0.2]) #dataset = 1
    # s = np.array([[wm_snr, 0, -csf_snr], [wm_snr, -gm_snr, 0], [0, gm_snr, -csf_snr] ]) #SNR matrix
    s = np.array([[wm_snr, -gm_snr, 0], [wm_snr, 0, -csf_snr], [0, gm_snr, -csf_snr] ]) #SNR matrix
    c = toy_c = s @known_m
    print("known_m = ", known_m, "toy_c: ",c)
    return s, c


class GridSearch :
  
  def __init__(self, s, c, upper_bound=1.0):
    self.param_m = [0.1, 0.15, 0.2, 0.25 , 0.3, 0.35 , 0.4, 0.5] #init M search space
    # self.param_m = [0.1, 0.2, 0.3, 0.4, 0.5]
    self.param_epsilon =  [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5] #regularization_strength search space
    self.solver = 'trust-constr'
    # self.limits = Bounds(0,1)
    self.limits = Bounds(0, upper_bound)
    self.losses = []
    self.results = {}
    self.results["m_init"] = []
    self.results["epsilon"] = []
    self.results["loss"] = []
    self.results["res"] = []
    self.s = s
    self.c = c

  def obj_function(self,m):
    loss = 0.5 * np.sum(((self.s@m) - self.c)**2)
    reg = (np.sum(m**2))
    # print(loss,reg)
    total_loss = loss + (self.epsilon * reg)
    self.losses.append(total_loss)
    return total_loss
  
  def easy_results(self, grid_results):
    grid_results["niter"] = []
    grid_results["x"] = []
    grid_results["cost"] = []

    for i in range(len(grid_results["res"])):
        res_obj = grid_results["res"][i]
        grid_results["niter"].append(res_obj.niter)
        grid_results["x"].append(res_obj.x)
        grid_results["cost"].append(res_obj.fun)
    df = pd.DataFrame.from_dict(grid_results)
    return df
  
  def solve(self):
    for i in self.param_m:
      for j in self.param_epsilon:
        self.m = np.ones(self.c.shape[0]) * i
        self.epsilon = j
        self.losses = []
        # print("Before optim", self.m)
        self.res = minimize(self.obj_function, self.m, bounds=self.limits, method=self.solver,)
        # print("After optim", self.res.x)
        self.results["m_init"].append(self.m)
        self.results["epsilon"].append(self.epsilon)
        self.results["loss"].append(self.losses)
        self.results["res"].append(self.res)
        
    return self.results
  
  def select_solution(self, grid_results):
    '''
    - Selects the 'best' solution from the subset of optimal solutions. This constraint is important as sometimes solver can find solutions where the multipliers are on different scales. 
    - Alternatively, 'best' solution can also be handpicked from the optimal subset


    Takes the grid results dataframe as input 
    Creates a temporary dataframe to filter the solutions based on minimal cost function and order of magnitude of solutions to be on same scale. 
    Returns the solution list (vector) : M
    '''
    temp_df = grid_results[grid_results['cost']<grid_results['cost'].min()+0.01]
    temp_df["log_constraint"] = " "

    for index in range(len(temp_df)):
        diff = abs(np.abs(np.floor(np.log10(((temp_df['x']).values[index][0])))) - np.abs(np.floor(np.log10(((temp_df['x']).values[index][1]))))) + abs(np.abs(np.floor(np.log10(((temp_df['x']).values[index][0])))) - np.abs(np.floor(np.log10(((temp_df['x']).values[index][2]))))) + abs(np.abs(np.floor(np.log10(((temp_df['x']).values[index][1])))) - np.abs(np.floor(np.log10(((temp_df['x']).values[index][2])))))
        temp_df['log_constraint'].values[index] = diff
    if(len(temp_df[temp_df['log_constraint']<1]) > 0):
        M = temp_df[temp_df['log_constraint']<1][temp_df[temp_df['log_constraint']<1]['cost']==temp_df[temp_df['log_constraint']<1]['cost'].min()]["x"].tolist()[0]
    else:
        #In case all the found solutions have different orders of magnitude, choose the minimal cost func. (Undesired case)
        M = grid_results[grid_results['log_constraint']==grid_results['log_constraint'].min()]["x"].tolist()[0]
    return M



def get_hf_observed_segmentations(dataset_num, config):
    folder = './Data/validation_data/sub_' + dataset_num +'/'
    hf_observed_nib, hf_wm_nib, hf_gm_nib, hf_csf_nib  = get_hf_observed(folder)
    (hf_wm_seg, hf_gm_seg, hf_csf_seg, hf_bg_seg) = get_tissue_seg(hf_wm_nib, hf_gm_nib, hf_csf_nib)
    # hf_observed_seg_dice = torch.stack((hf_bg_seg[0].reshape(config["size_lf"]), hf_wm_seg[0].reshape(config["size"]), hf_gm_seg[0].reshape(config["size"]), hf_csf_seg[0].reshape(config["size"])), dim=0).unsqueeze(0)
    return (hf_wm_seg, hf_gm_seg, hf_csf_seg, hf_bg_seg)

def get_lf_observed_segmentations(dataset_num, config):
    folder = './Data/validation_data/sub_' + dataset_num +'/'
    (img_nib, wm_nib, gm_nib, csf_nib) = read_ulf_imgs(folder) #return ULF data 
    (wm_seg, gm_seg, csf_seg, bg_seg) = get_tissue_seg(wm_nib, gm_nib, csf_nib)
    lf_wm_seg, lf_gm_seg, lf_csf_seg, lf_bg_seg = (torch.from_numpy(wm_seg), torch.from_numpy(gm_seg), torch.from_numpy(csf_seg), torch.from_numpy(bg_seg))
    # lf_observed_seg_dice = torch.stack((lf_bg_seg[0].reshape(config["size_lf"]), lf_wm_seg[0].reshape(config["size_lf"]), lf_gm_seg[0].reshape(config["size_lf"]), lf_csf_seg[0].reshape(config["size_lf"])), dim=0).unsqueeze(0)
    return lf_wm_seg.flatten().to(torch.float32).unsqueeze(0), lf_gm_seg.flatten().to(torch.float32).unsqueeze(0), lf_csf_seg.flatten().to(torch.float32).unsqueeze(0), lf_bg_seg.flatten().to(torch.float32).unsqueeze(0)


def forward(dataset_num='0011'):
    '''
    This function will generate the ground truth data for a given HF Image
    '''
    
    folder = './Data/validation_data/sub_' + dataset_num + '/'

    (ulf_img_nib, ulf_wm_nib, ulf_gm_nib, ulf_csf_nib) = read_ulf_imgs(folder) #return ULF data
    (ulf_wm_seg, ulf_gm_seg, ulf_csf_seg, ulf_bg_seg) = get_tissue_seg(ulf_wm_nib, ulf_gm_nib, ulf_csf_nib)
    (ulf_wm, ulf_gm, ulf_csf, ulf_bg, hf) = seg_to_intenities(ulf_img_nib, ulf_wm_nib, ulf_gm_nib, ulf_csf_nib, ulf_bg_seg)
    # (wm_snr, gm_snr, csf_snr) = calc_val_ulf_snr(img_nib, wm_nib, gm_nib, csf_nib, dataset_num)

    # s, c = toy_values(wm_snr, gm_snr, csf_snr, dataset_num)
    # # c = np.array([9.03, 27.17, 18.14, ]) #this is ULF contrast vector
    # # c = np.array([13.47, 39.52, 26.05]) #this is HF contrast vector #from ULFEnc dataset
    # c = np.array([96.026, 343.63, 247.604])
    # print("SNR matrix:", s, "Target Contrast: " , c)
    # grid = GridSearch(s,c)
    # grid_results = grid.solve()
    # grid_results = grid.easy_results(grid_results)
    # print(grid_results)
    # M = grid_results.iloc[0].x
    # M = grid.select_solution(grid_results)
    s, target_c, M = get_m('0011', target_type='hf')
    print("Solution : ", M)
    
    print("Target Contrast: " , target_c, "Achieved contrast: ", s@M)

    hf_observed_nib, hf_wm_nib, hf_gm_nib, hf_csf_nib  = get_hf_observed(folder)
    (hf_wm_seg, hf_gm_seg, hf_csf_seg, hf_bg_seg) = get_tissue_seg(hf_wm_nib, hf_gm_nib, hf_csf_nib)
    lf_observed = (ulf_img_nib.get_fdata())
    hf_observed = hf_observed_nib.get_fdata()
    # hf_observed_seg_dice = torch.stack((hf_bg_seg[0].reshape(config["size_lf"]), hf_wm_seg[0].reshape(config["size"]), hf_gm_seg[0].reshape(config["size"]), hf_csf_seg[0].reshape(config["size"])), dim=0).unsqueeze(0)
    # lf_wm_seg, lf_gm_seg, lf_csf_seg, lf_bg_seg = wm_nib.get_fdata(), gm_nib.get_fdata(), csf_nib.get_fdata(), bg_nib.get_fdata()  #Load ULF Observed Segmentations
    # lf_observed_seg_dice = torch.stack((lf_bg_seg[0].reshape(config["size_lf"]), lf_wm_seg[0].reshape(config["size_lf"]), lf_gm_seg[0].reshape(config["size_lf"]), lf_csf_seg[0].reshape(config["size_lf"])), dim=0).unsqueeze(0)
    # M = [1, 1, 1 ]
    return hf_observed, lf_observed, M
    
    
def load_val_data(dataset_num, config=default_config):
    
    slice = config["slice"] 
    #Load HF, ULF observed
    
    hf_observed, lf_observed, M = forward(dataset_num) #Generating ULF observed
    config["M"] = M
    lf_observed = norm(lf_observed) #normalized
    
    #Updating size config parameters
    config["size"] = (hf_observed.shape[0], hf_observed.shape[1], hf_observed.shape[2]) #Loading 2D Images. For datasets = 1, 2 (174, 192)
    config["size_lf"] = (lf_observed.shape[0], lf_observed.shape[1], lf_observed.shape[2])
    
 
    lf_wm_seg, lf_gm_seg, lf_csf_seg, lf_bg_seg = get_lf_observed_segmentations(dataset_num, config) #Load ULF Observed Segmentations
    lf_observed_seg_dice = torch.stack((lf_bg_seg[0].reshape(config["size_lf"]), lf_wm_seg[0].reshape(config["size_lf"]), lf_gm_seg[0].reshape(config["size_lf"]), lf_csf_seg[0].reshape(config["size_lf"])), dim=0).unsqueeze(0)
 

    # return hf_observed, lf_dataloader, lf_observed, M #might need to uncomment this. lf_dataloader is the default used everywhere
    return hf_observed, lf_observed, lf_observed_seg_dice, M #might need to comment this


