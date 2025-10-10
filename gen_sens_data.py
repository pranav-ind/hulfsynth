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



dataset_num = 102
folder = './Data/ixi/T1/' + str(dataset_num) + "/"
(img_nib, wm_nib, gm_nib, csf_nib) = read_imgs(folder)
(wm_seg, gm_seg, csf_seg, bg_seg) = get_hf_tissue_seg(wm_nib, gm_nib, csf_nib)
(wm, gm, csf, bg, hf) = seg_to_intenities(img_nib, wm_nib, gm_nib, csf_nib, bg_seg)
# (wm_snr, gm_snr, csf_snr) = calc_hf_snr(img_nib, wm_nib, gm_nib, csf_nib, dataset_num)



