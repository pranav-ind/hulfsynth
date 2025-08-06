import torch
import pandas as pd
import numpy as np
import torchvision.transforms as Tr
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
# from tkinter import filedialog as fd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import nibabel as nib
import matplotlib as mpl
from skimage.exposure import match_histograms
# from nilearn import image
# from nilearn import plotting
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


img_loc = './data_1/T1_brain.nii.gz'
output_folder = './data_1/'

type = 1 #type of image 1=T1, 2=T2, 3=PD; default=T1
'''
fast(img_loc, out= output_folder+'fast', g= True, B=True, n_classes=3, t=1)
print("FSL Fast Executed")
'''



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


def read_imgs(output_folder):
    output_folder = output_folder + "fast"
    csf_nib = nib.load(output_folder + "_pve_0.nii.gz") 
    gm_nib = nib.load(output_folder + "_pve_1.nii.gz") 
    wm_nib = nib.load(output_folder + "_pve_2.nii.gz") 
    img_nib = nib.load(output_folder + "_restore.nii.gz")
    return (img_nib, wm_nib, gm_nib, csf_nib)

def read_imgs2(output_folder):
    output_folder = output_folder + "structural_brain"
    csf_nib = nib.load(output_folder + "_pve_0.nii.gz") 
    gm_nib = nib.load(output_folder + "_pve_1.nii.gz") 
    wm_nib = nib.load(output_folder + "_pve_2.nii.gz") 
    img_nib = nib.load(output_folder + "_restore.nii.gz")
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


def plot_4_images(wm, gm, csf, hf, slice = 95, vmax = [100,200,400, 300], titles = ["White Matter", "Gray Matter", "CSF", "High Field Image"], fig_title="Segmentations (Intensitites)"):
    # slice = 95
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    im1 = axs[0].imshow(wm[:,:,slice],cmap='gray',vmax = vmax[0])
    im2 = axs[1].imshow(gm[:,:,slice],cmap='gray',vmax = vmax[1])
    im3 = axs[2].imshow(csf[:,:,slice],cmap='gray' ,vmax = vmax[2])
    im4 = axs[3].imshow((hf)[:,:,slice],cmap='gray',vmax = vmax[3])


    axs[0].set_title(titles[0])
    axs[1].set_title(titles[1])
    axs[2].set_title(titles[2])
    axs[3].set_title(titles[3])


    fig.colorbar(im1, ax=axs[0],shrink=0.5)
    fig.colorbar(im2, ax=axs[1],shrink=0.5)
    fig.colorbar(im3, ax=axs[2],shrink=0.5)
    fig.colorbar(im4, ax=axs[3],shrink=0.5)

    fig.suptitle(fig_title, fontsize=14,color='blue')
    plt.show()



#Functions specific to dataset. #Bad coding practice. Refactor after deadline

def plot_rois(wm, gm, csf):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    im1 = axs[0].imshow(csf[:,:,95],cmap='gray')# ,vmax = 1000)
    im2 = axs[1].imshow(gm[:,:,95],cmap='gray')#,vmax = 800)
    im3 = axs[2].imshow(wm[:,:,95],cmap='gray')#,vmax = 600)


    # rect3 = patches.Rectangle((55, 69), 5, 3, linewidth=1, edgecolor='r', facecolor='none') #ms 
    rect2 = patches.Rectangle((145, 99), 10, 10, linewidth=1, edgecolor='r', facecolor='none') #wm 
    rect1 = patches.Rectangle((123, 76), 5, 4, linewidth=0.7, edgecolor='r', facecolor='none') #gm 
    rect0 = patches.Rectangle((119, 82), 9, 4, linewidth=1, edgecolor='r', facecolor='none') #csf 

    axs[0].add_patch(rect0)
    axs[1].add_patch(rect1)
    axs[2].add_patch(rect2)
    # axs[3].add_patch(rect3)

    fig.suptitle('ROI for SNR calculation', fontsize=14,color='blue')
    plt.show()



def calc_hf_snr(img_nib, wm_nib, gm_nib, csf_nib):
    #for dataset =1
    rayleigh_correction = 1.53

    noise = nib.load("./data_1/T1.nii.gz")
    print("BG Noise in different regions :", noise.get_fdata()[:20,:20,95].std(), noise.get_fdata()[150:175,150:175,95].std(), noise.get_fdata()[155:,:25,95].std(), noise.get_fdata()[:25,150:170,95].std()) #Mean of all 4 regions in background
    std_bg = noise.get_fdata()[:20,:20,95].std() #Background Noise extracted manually


    wm_roi_pixels = img_nib.get_fdata()[99:109,145:155,95][wm_nib.get_fdata()[99:109,145:155,95]>0.9]
    gm_roi_pixels = img_nib.get_fdata()[76:80, 123:128,95][gm_nib.get_fdata()[76:80, 123:128,95]>0.9]
    csf_roi_pixels = img_nib.get_fdata()[82:86,119:128,95][csf_nib.get_fdata()[82:86,119:128,95]>0.9]

    csf_snr = csf_roi_pixels.mean()/ (std_bg * rayleigh_correction)
    gm_snr = gm_roi_pixels.mean()/ (std_bg * rayleigh_correction)
    wm_snr = wm_roi_pixels.mean()/ (std_bg * rayleigh_correction)


    Ccg = abs(csf_snr - gm_snr) 
    Ccw = abs(csf_snr - wm_snr)
    Cgw = abs(gm_snr - wm_snr)

    return (wm_snr, gm_snr, csf_snr)

def calc_hf_snr2(img_nib, wm_nib, gm_nib, csf_nib):
    #for dataset = 2 
    rayleigh_correction = 1.53

    #Noise for dataset = 2
    # noise = nib.load("./data_2/structural.nii.gz")
    noise = nib.load("./structural.nii.gz") #Temporarily running in the same folder
    print("BG Noise in different regions :", noise.get_fdata()[:20,:20,95].std(), noise.get_fdata()[150:170,150:170,95].std(), noise.get_fdata()[155:,:25,95].std(), noise.get_fdata()[:25,150:170,95].std()) #Mean of all 4 regions in background
    std_bg = noise.get_fdata()[:20,:20,95].std() #Background Noise extracted manually


    wm_roi_pixels = img_nib.get_fdata()[105:110, 132:142, 95][wm_nib.get_fdata()[105:110, 132:142, 95]>0.9]
    gm_roi_pixels = img_nib.get_fdata()[45:48,43:48,95][gm_nib.get_fdata()[45:48,43:48,95]>0.9]
    csf_roi_pixels = img_nib.get_fdata()[96:101,70:80,95][csf_nib.get_fdata()[96:101,70:80,95]>0.9]

    csf_snr = csf_roi_pixels.mean()/ (std_bg * rayleigh_correction)
    gm_snr = gm_roi_pixels.mean()/ (std_bg * rayleigh_correction)
    wm_snr = wm_roi_pixels.mean()/ (std_bg * rayleigh_correction)


    Ccg = abs(csf_snr - gm_snr) 
    Ccw = abs(csf_snr - wm_snr)
    Cgw = abs(gm_snr - wm_snr)

    return (wm_snr, gm_snr, csf_snr)





def toy_values(wm_snr, gm_snr, csf_snr):
    # known_m = np.array([0.6, 0.6, 0.3])
    # known_m = np.array([0.7, 0.3, 0.2])
    known_m = np.array([0.5, 0.6, 0.2])
    s = np.array([[wm_snr, 0, -csf_snr], [wm_snr, -gm_snr, 0], [0, gm_snr, -csf_snr] ]) #SNR matrix
    c = toy_c = s @known_m
    return s, c


class GridSearch :
  
  def __init__(self, s, c):
    self.param_m = [0.1, 0.15, 0.2, 0.25 , 0.3, 0.35 , 0.4, 0.5]
    # self.param_m = [0.1, 0.2, 0.3, 0.4, 0.5]
    self.param_epsilon =  [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
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
    Takes the grid results dataframe as input 
    Creates a temporary dataframe to filter the solutions based on minimal cost function and order of magnitude of solutions to be on same scale. 
    Returns the solution list (vector) : M
    '''
    temp_df = grid_results[grid_results['cost']<grid_results['cost'].min()+0.01]
    temp_df["log_constraint"] = " "
    # for index, row in temp_df.iterrows():
    for index in range(len(temp_df)):
        # print(index)
        diff = abs(np.abs(np.floor(np.log10(((temp_df['x']).values[index][0])))) - np.abs(np.floor(np.log10(((temp_df['x']).values[index][1]))))) + abs(np.abs(np.floor(np.log10(((temp_df['x']).values[index][0])))) - np.abs(np.floor(np.log10(((temp_df['x']).values[index][2]))))) + abs(np.abs(np.floor(np.log10(((temp_df['x']).values[index][1])))) - np.abs(np.floor(np.log10(((temp_df['x']).values[index][2])))))
        # temp_df.at[index, 'log_constraint'] = diff
        temp_df['log_constraint'].values[index] = diff
    if(len(temp_df[temp_df['log_constraint']<1]) > 0):
        M = temp_df[temp_df['log_constraint']<1][temp_df[temp_df['log_constraint']<1]['cost']==temp_df[temp_df['log_constraint']<1]['cost'].min()]["x"].tolist()[0]
    else:
        #In case all the found solutions have different orders of magnitude, choose the minimal 
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





def forward(dataset_folder = "./data_1/"):
    '''
    This function will generate the ground truth data for a given HF Image
    '''


    
    img_nib = dataset_folder + "T1_brain.nii.gz"
    (img_nib, wm_nib, gm_nib, csf_nib) = read_imgs(dataset_folder)
    (wm_prob, gm_prob, csf_prob, bg_prob) = tissue_probabilities(wm_nib, gm_nib, csf_nib)
    (wm, gm, csf, bg, hf) = seg_to_intenities(img_nib, wm_nib, gm_nib, csf_nib, bg_prob)
    # plot_4_images(wm,gm,csf, hf)
    # plot_rois(wm,gm,csf)
    (wm_snr, gm_snr, csf_snr) = calc_hf_snr(img_nib, wm_nib, gm_nib, csf_nib)
    s, c = toy_values(wm_snr, gm_snr, csf_snr)
    grid = GridSearch(s,c)
    grid_results = grid.solve()
    grid_results = grid.easy_results(grid_results)
    M = grid.select_solution(grid_results)
    print("Solution : ", M)
    (wm_lf_like, gm_lf_like, csf_lf_like, bg_lf_like, lf_like) = recombine(wm, gm, csf, bg, M)
    lf_like = lf_like + np.random.normal(1, 0.5, size=lf_like.shape)
    return (wm_lf_like, gm_lf_like, csf_lf_like, bg_lf_like, lf_like), (wm_prob, gm_prob, csf_prob, bg_prob), (wm_snr, gm_snr, csf_snr), M
    # plot_4_images(wm_lf_like, gm_lf_like, csf_lf_like, lf_like,vmax=[100, 100, 100, 100])
    # print(M)
    # print(grid_results.head())


def forward2(dataset_folder = "./data_1/", dataset=1):
    '''
    This function will generate the ground truth data for a given HF Image
    '''
    
    dataset_folder = "./" #Temporarily running from the same folder
    img_nib = dataset_folder + "structural_brain.nii.gz"
    (img_nib, wm_nib, gm_nib, csf_nib) = read_imgs2(dataset_folder)
    (wm_prob, gm_prob, csf_prob, bg_prob) = tissue_probabilities(wm_nib, gm_nib, csf_nib)
    (wm, gm, csf, bg, hf) = seg_to_intenities(img_nib, wm_nib, gm_nib, csf_nib, bg_prob)
    (wm_snr, gm_snr, csf_snr) = calc_hf_snr2(img_nib, wm_nib, gm_nib, csf_nib)



    # plot_4_images(wm,gm,csf, hf)
    # plot_rois(wm,gm,csf)

    s, c = toy_values(wm_snr, gm_snr, csf_snr)
    print(s,c)
    grid = GridSearch(s,c)
    grid_results = grid.solve()
    grid_results = grid.easy_results(grid_results)
    M = grid.select_solution(grid_results)
    print("Solution : ", M)
    (wm_lf_like, gm_lf_like, csf_lf_like, bg_lf_like, lf_like) = recombine(wm, gm, csf, bg, M)
    lf_like = lf_like + np.random.normal(1, 0.5, size=lf_like.shape)
    # plot_4_images(wm_lf_like, gm_lf_like, csf_lf_like, lf_like)#,vmax=[100, 100, 100, 100])
    return (wm_lf_like, gm_lf_like, csf_lf_like, bg_lf_like, lf_like), (wm_prob, gm_prob, csf_prob, bg_prob), (wm_snr, gm_snr, csf_snr), M
    

    
    
forward2()

# main()









'''
start = time.time()
segment(img_loc, output_folder)
end = time.time()
print(end-start)
'''