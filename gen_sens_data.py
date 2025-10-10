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
import ast

from LFSynth.ContrastEstimation import read_imgs, get_hf_tissue_seg, seg_to_intenities, calc_hf_snr, toy_values, GridSearch, add_rician, recombine 


def get_rois(dataset_num='0011', field_type='hf'):
    folder = './Data/ixi/T1/' + str(dataset_num) +  '/' + field_type
    file_path =  folder + '/snrs.txt'
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

def get_m(dataset_num='0011', target_type='hf', target_c = np.array([10.62876390679262, 36.76892223997362, 26.140158333180995])):
    #m^{ULF} = get_m (c^{HF}, S^{ULF}) i.e., if target_type == ULF then S matrix should be from HF and if target_type == HF then S matrix should be ULF ;
    if(target_type == 'ulf'):
        S_type = 'hf'
        upper_bound = 1.0
    else:
        S_type = 'ulf'
        upper_bound = np.inf
    rayleigh_correction = 1.53
    _, S_list = get_rois(dataset_num, field_type=S_type)
    # print("S list", S_list)
    s = np.array([[S_list[0], -S_list[1], 0], [S_list[0], 0, -S_list[2]], [0, S_list[1], -S_list[2]]]) #SNR matrix #S_list[0] = WM, S_list[1] = GM, S_list[2] = CSF
    # target_c = get_target_c(dataset_num, target_type)
    # target_c = np.array([19.59, 74.456, 54.86])
    # target_c = np.array([10.62876390679262, 36.76892223997362, 26.140158333180995])
    grid = GridSearch(s,target_c, upper_bound=upper_bound)
    grid_results = grid.solve()
    grid_results = grid.easy_results(grid_results)
    # print(grid_results)
    M = grid.select_solution(grid_results)
    
    # M = grid_results.iloc[0].x
    return s, target_c, M


dataset_num = 102
folder = './Data/ixi/T1/' + str(dataset_num) + "/"
write_folder = './Data/ixi/T1/' + str(dataset_num) + '/sensitivity_data/contrast/'

'''
(img_nib, wm_nib, gm_nib, csf_nib) = read_imgs(folder)
(wm_seg, gm_seg, csf_seg, bg_seg) = get_hf_tissue_seg(wm_nib, gm_nib, csf_nib)
(wm, gm, csf, bg, hf) = seg_to_intenities(img_nib, wm_nib, gm_nib, csf_nib, bg_seg)
# (wm_snr, gm_snr, csf_snr) = calc_hf_snr(img_nib, wm_nib, gm_nib, csf_nib, dataset_num)

c_list = (
np.array([19.591889626325028, 74.45619024020756, 54.86430061388253]),
np.array([13.309712111059078, 35.49778633247732, 22.18807422141824]),
np.array([10.62876390679262, 36.76892223997362, 26.140158333180995]),
np.array([12.08320064326, 38.80917554433573, 26.72597490107573]),
np.array([15.131863348522124, 57.951366902350586, 42.81950355382846]))

for i in range(len(c_list)):
    s, target_c, M = get_m(dataset_num, target_type='ulf', target_c = c_list[i])
    print("SNR matrix:", s, "Target Contrast: " , target_c, "Solution : ", M)
    print("Target Contrast: " , target_c, "Achieved contrast: ", s@M)
    (wm_lf_like, gm_lf_like, csf_lf_like, bg_lf_like, lf_like) = recombine(wm, gm, csf, bg, M)
    mask = np.where(lf_like>0 ,1.0, 0.0)
    rician_noise = add_rician(lf_like.shape, v = 5, s = 15)
    ulf_like = lf_like + (rician_noise * mask) #adding rician noise only to foreground voxels
    ulf_nib = nib.Nifti1Image(ulf_like, np.eye(4)) #without bg noise
    file_name = write_folder + str(i+1) + '/brain.nii.gz'
    nib.save(ulf_nib, file_name)
    print('Saving nib file to', file_name, '.....')
'''



'''
#Segmenting images
for i in range(5):
    folder_loc = write_folder + str(i+1) + '/'
    fast(folder_loc + 'brain.nii.gz', out= folder_loc + 'fast', g= False, b = True , B = True, n_classes=3, t=1)
    print("segmented ", str(i), '.....')
'''
