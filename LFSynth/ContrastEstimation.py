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


def read_imgs(folder):
    #Expected Folder Name: "./data_" + str(dataset_num) + "/". Returns ground truth image and segmentations in nib format
    folder = folder + "fast"
    csf_nib = nib.load(folder + "_pve_0.nii.gz") 
    gm_nib = nib.load(folder + "_pve_1.nii.gz") 
    wm_nib = nib.load(folder + "_pve_2.nii.gz") 
    img_nib = nib.load(folder + "_restore.nii.gz")
    return (img_nib, wm_nib, gm_nib, csf_nib)


def tissue_probabilities(wm_nib, gm_nib, csf_nib):
    '''
    Adds up all probability segmentations and creates a new segmentation for background tissue. Expected : sum of all (4) tissue probabilities = 1
    Returns a tuple of probabilities
    '''
    csf =  csf_nib.get_fdata()  
    gm =   gm_nib.get_fdata() 
    wm =   wm_nib.get_fdata()
    total_prob = csf + gm + wm
    total_prob = total_prob.clip(max=1) #Because for some voxels the summation is giving value slightly greater than 1.0 (ex : 1.0000000298023224). So clipping max value to 1
    bg = 1 - total_prob
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


def calc_hf_snr(img_nib, wm_nib, gm_nib, csf_nib, dataset):

    #for dataset = 1 and 2 only
    rayleigh_correction = 1.53

    if(dataset ==1):
        #ROIs are extracted manually. For example, refer roi_gen.ipynb 
        wm_roi_pixels = img_nib.get_fdata()[99:109,145:155,95][wm_nib.get_fdata()[99:109,145:155,95]>0.9]
        gm_roi_pixels = img_nib.get_fdata()[76:80, 123:128,95][gm_nib.get_fdata()[76:80, 123:128,95]>0.9]
        csf_roi_pixels = img_nib.get_fdata()[82:86,119:128,95][csf_nib.get_fdata()[82:86,119:128,95]>0.9]

    else:
        #ROIs are extracted manually. For example, refer roi_gen.ipynb 
        wm_roi_pixels = img_nib.get_fdata()[105:110, 132:142, 95][wm_nib.get_fdata()[105:110, 132:142, 95]>0.9]
        gm_roi_pixels = img_nib.get_fdata()[45:48,43:48,95][gm_nib.get_fdata()[45:48,43:48,95]>0.9]
        csf_roi_pixels = img_nib.get_fdata()[96:101,70:80,95][csf_nib.get_fdata()[96:101,70:80,95]>0.9] 

    noise_file = "./Data/data_" + str(dataset) + "/raw.nii.gz"
    noise = nib.load(noise_file)
    print("BG Noise in different regions :", noise.get_fdata()[:20,:20,95].std(), noise.get_fdata()[150:175,150:175,95].std(), noise.get_fdata()[155:,:25,95].std(), noise.get_fdata()[:25,150:170,95].std()) #Mean of all 4 regions in background
    std_bg = noise.get_fdata()[:20,:20,95].std() #Background Noise extracted manually


    csf_snr = csf_roi_pixels.mean()/ (std_bg * rayleigh_correction)
    gm_snr = gm_roi_pixels.mean()/ (std_bg * rayleigh_correction)
    wm_snr = wm_roi_pixels.mean()/ (std_bg * rayleigh_correction)


    Ccg = abs(csf_snr - gm_snr) 
    Ccw = abs(csf_snr - wm_snr)
    Cgw = abs(gm_snr - wm_snr)

    return (wm_snr, gm_snr, csf_snr)


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
    s = np.array([[wm_snr, 0, -csf_snr], [wm_snr, -gm_snr, 0], [0, gm_snr, -csf_snr] ]) #SNR matrix
    c = toy_c = s @known_m
    print("known_m = ", known_m)
    return s, c


class GridSearch :
  
  def __init__(self, s, c):
    self.param_m = [0.1, 0.15, 0.2, 0.25 , 0.3, 0.35 , 0.4, 0.5] #init M search space
    # self.param_m = [0.1, 0.2, 0.3, 0.4, 0.5]
    self.param_epsilon =  [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5] #regularization_strength search space
    self.solver = 'trust-constr'
    self.limits = Bounds(0,1)
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




def downsample(img,sigma = 0.4):
    smooth_anat_img = gaussian_filter(img, sigma,mode='nearest')
    downsampled = smooth_anat_img[::2, ::2,:]
    return downsampled


def recombine(wm, gm, csf, bg, M):
    #Downsample, Contrast Modulation and Recombination on downsampled tissues
    
    csf_new = M[2] * downsample(csf)
    gm_new = M[1] * downsample(gm)
    wm_new = M[0] * downsample(wm)
    

    bg_new = downsample(bg)
    lf_img = csf_new + gm_new + wm_new + bg_new
    return wm_new, gm_new, csf_new, bg_new, lf_img

def add_rician(size_lf, v = 2, s = 3):
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
    return noise.reshape(size_lf)


def forward(dataset_num=1):
    '''
    This function will generate the ground truth data for a given HF Image
    '''
    
    folder = "./Data/data_" + str(dataset_num) + "/"
    (img_nib, wm_nib, gm_nib, csf_nib) = read_imgs(folder)
    (wm_prob, gm_prob, csf_prob, bg_prob) = tissue_probabilities(wm_nib, gm_nib, csf_nib)
    (wm, gm, csf, bg, hf) = seg_to_intenities(img_nib, wm_nib, gm_nib, csf_nib, bg_prob)
    (wm_snr, gm_snr, csf_snr) = calc_hf_snr(img_nib, wm_nib, gm_nib, csf_nib, dataset_num)

    s, c = toy_values(wm_snr, gm_snr, csf_snr, dataset_num)
    print(s,c)
    grid = GridSearch(s,c)
    grid_results = grid.solve()
    grid_results = grid.easy_results(grid_results)
    print(grid_results)
    M = grid.select_solution(grid_results)
    print("Solution : ", M)
    (wm_lf_like, gm_lf_like, csf_lf_like, bg_lf_like, lf_like) = recombine(wm, gm, csf, bg, M)
    # lf_like = lf_like + np.random.normal(2, 0.75, size=lf_like.shape) #adding gaussian noise
    lf_like = lf_like + add_rician(lf_like.shape) #adding rician noise
    


    # plot_4_images(wm_lf_like, gm_lf_like, csf_lf_like, lf_like)#,vmax=[100, 100, 100, 100])
    return (wm_lf_like, gm_lf_like, csf_lf_like, bg_lf_like, lf_like), (wm_prob, gm_prob, csf_prob, bg_prob), (wm_snr, gm_snr, csf_snr), M


# forward(dataset_num = 1)