import nibabel as nib
import torch, torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import os
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from monai.losses.dice import *  # NOQA
from monai.losses.dice import DiceLoss, DiceCELoss
from IPython.display import clear_output
from monai import metrics
import tempfile, os
import monai
from monai.losses import DiceLoss, DiceCELoss, SoftclDiceLoss, DiceFocalLoss, NACLLoss
from torch.nn import KLDivLoss
import kornia
import os
import random
from torchmetrics.functional.image import image_gradients
from LFSynth.ContrastEstimation import forward as contrast_forward
from Utils.utils import get_device, get_metrics, pad_chunk


class ContrastModulation:
  #3D
  
  def __init__(self, ):
    self.size = (87*2,96*2)

  def downsample(self, img):
      
      img = img.unsqueeze(0)
      img = torchvision.transforms.functional.gaussian_blur(img, kernel_size=[3,3])
      img = img[:, ::2, ::2]
      return img.squeeze(0)
  
  def forward(self, chunk_size, model_output_seg, model_output_img, M):
    #for chunks #expected input shape = (B, C, H*W*D, 4)
    # size = (1, 1, 87, 96, 96)
    # size_lf = (87, 96, 96)
    # size_hf = (87*2, 96*2, 96*2)
    # chunk_size = (43*2,48*2,48*2)
    # chunk_size = (43*2,48*2,48*2)
    model_output_seg = model_output_seg.squeeze(0)
    model_output_img = model_output_img.squeeze(0)

    # model_output_seg = pad_chunk(model_output_seg)
    # model_output_img = pad_chunk(model_output_img)
    
    #Weighted Sum of Segmentation Probabilities and Image Intensities
    # wm_img = (model_output_img[:,:,0] * model_output_seg[:,:,0])
    # gm_img = (model_output_img[:,:,1] * model_output_seg[:,:,1])
    # csf_img = (model_output_img[:,:,2] * model_output_seg[:,:,2])
    # bg_img = (model_output_img[:,:,3] * model_output_seg[:,:,3])
    # print("WM Img Shape = ", wm_img.shape)

    wm_img = (model_output_img[:,0] * model_output_seg[:,0]).reshape(chunk_size) #shape : *spatial,4 to shape : (chunk size)
    gm_img = (model_output_img[:,1] * model_output_seg[:,1]).reshape(chunk_size)
    csf_img = (model_output_img[:,2] * model_output_seg[:,2]).reshape(chunk_size)
    bg_img = (model_output_img[:,3] * model_output_seg[:,3]).reshape(chunk_size)
    
    '''
    wm_img = pad_chunk(wm_img, chunk_size)
    gm_img = pad_chunk(gm_img, chunk_size)
    csf_img = pad_chunk(csf_img, chunk_size)
    bg_img = pad_chunk(bg_img, chunk_size) #padding to (44,48,48)
    '''
    
    # gm_img = pad1(gm_img).squeeze(0).squeeze(0)
    # csf_img = pad1(csf_img).squeeze(0).squeeze(0)
    # bg_img = pad1(bg_img).squeeze(0).squeeze(0)

    #Downsample and Contrast Modulation 
    
    csf_img = M[2] * (csf_img[:,::2, ::2])
    gm_img = M[1] * (gm_img[:,::2, ::2])
    wm_img = M[0] * (wm_img[:,::2, ::2])
    bg_img = (bg_img[:,::2, ::2]) #downsampling to (48,22,24)
    

    # Recombination of downsampled tissues 
    lf_img = csf_img + gm_img + wm_img + bg_img
    return lf_img.flatten().unsqueeze(0).unsqueeze(-1) #(1,*spatial,1)


# con_mod = ContrastModulation()







"""
#  used for 2D
class ContrastModulation:
  def __init__(self, ):
    self.size = (87*2,96*2)

  def downsample(self, img):
      
      img = img.unsqueeze(0)
      img = torchvision.transforms.functional.gaussian_blur(img, kernel_size=[3,3])
      img = img[:, ::2, ::2]
      return img.squeeze(0)
  
  def forward(self, model_output_seg, model_output_img, M):
    size = self.size
      
    
    #Weighted Sum of Segmentation Probabilities and Image Intensities
    # wm_img = (model_output_img[:,:,0] * model_output_seg[:,:,0])
    # gm_img = (model_output_img[:,:,1] * model_output_seg[:,:,1])
    # csf_img = (model_output_img[:,:,2] * model_output_seg[:,:,2])
    # bg_img = (model_output_img[:,:,3] * model_output_seg[:,:,3])
    # print("WM Img Shape = ", wm_img.shape)

    wm_img = (model_output_img[:,:,0] * model_output_seg[:,:,0]).reshape(size)
    gm_img = (model_output_img[:,:,1] * model_output_seg[:,:,1]).reshape(size)
    csf_img = (model_output_img[:,:,2] * model_output_seg[:,:,2]).reshape(size)
    bg_img = (model_output_img[:,:,3] * model_output_seg[:,:,3]).reshape(size)
    

    # wm_img = (model_output_img * model_output_seg[:,:,0]).reshape(size)
    # gm_img = (model_output_img * model_output_seg[:,:,1]).reshape(size)
    # csf_img = (model_output_img * model_output_seg[:,:,2]).reshape(size)
    # bg_img = (model_output_img * model_output_seg[:,:,3]).reshape(size)

    #Downsample and Contrast Modulation 
    csf_img = M[2] * (csf_img[::2, ::2])
    gm_img = M[1] * (gm_img[::2, ::2])
    wm_img = M[0] * (wm_img[::2, ::2])
    bg_img = (bg_img[::2, ::2])
    

    # Recombination of downsampled tissues 
    lf_img = csf_img + gm_img + wm_img + bg_img
    return lf_img.flatten().unsqueeze(-1).unsqueeze(0)

  """