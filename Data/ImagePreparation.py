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

import tempfile, os
import monai
# from crfseg import CRF


import kornia
import os
import random

from Data.FFE import GaussianFourierFeatureTransform





#Modified from SIREN example for non-square images. 
# TODO : Cite SIREN paper


#Modified from SIREN example for non-square images. 
# TODO : Cite SIREN paper

class ImagePreparation2D(Dataset):
  '''A class which takes as input a B/W image and preprocess it into a list of coords and pixels of the same image cropped and resized into (h,w)
  '''
  def __init__(self, img, h,w, is_ffe=False):
      super().__init__()

    #   img = Image.open(name)  
      img = img      
      transform = Compose([
      Resize((h,w)),
      ToTensor(),
      # Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
      ])
      # img = transform(img)
      img = torch.from_numpy(img)

      # self.pixels = img.permute(1, 2, 0).view(-1, 1)
      if(is_ffe):
        self.coords = self.get_mgrid2(h*2, w*2, 2) 
      else:
        self.coords = self.get_mgrid(h*2, w*2, 2)
      self.pixels = img.flatten().unsqueeze(0).unsqueeze(-1) #Shape = (1, size_lf[0]*size_lf[1], 1)
      # self.coords = self.get_mgrid(h*2, w*2, 2)
      
  def __len__(self):
      return 1

  def __getitem__(self, idx):    
      if idx > 0: raise IndexError
          
      return self.coords, self.pixels
  
  def get_mgrid(self, h,w, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    # tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    x_coords = torch.linspace(-1, 1, w)
    y_coords = torch.linspace(-1, 1, h)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='xy')
    mgrid = torch.stack((x_grid.flatten(), y_grid.flatten()), dim=-1)
    # mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid
  
  def get_mgrid2(self, h,w, dim=2):
    #Grid Generation for FFE
    '''Generates a grid of (x,y,...) coordinates in a range of -1 to 1.
    h,w = Height, Width
    dim: int'''
    x_coords = torch.linspace(-1, 1, w)
    y_coords = torch.linspace(-1, 1, h)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='xy')
    mgrid = torch.stack((x_grid, y_grid), dim=-1)
    mapping_size = 128
    num_channels = 2
    
    x = GaussianFourierFeatureTransform(num_channels, mapping_size, 5)(mgrid.permute(2, 0,1).unsqueeze(0))
    
    x = x.reshape(num_channels*mapping_size, w*h)
    x = x.permute(1, 0)
    return x
  

  #Testing
'''
orig_img = Image.fromarray(lf_gt)
test1 = ImagePreparation(orig_img, 87, 96)
print(test1.pixels.shape,test1.coords.shape)
'''  
  

  #Testing
'''
orig_img = Image.fromarray(lf_gt)
test1 = ImagePreparation(orig_img, 87, 96)
print(test1.pixels.shape,test1.coords.shape)
'''  






class ImagePreparation(Dataset):
    #Adapted from SIREN example for non-square images and added FFE Encoding
    # TODO : Cite SIREN paper
  '''A class which takes as input a B/W image and preprocess it into a list of coords and pixels of the same image cropped and resized into (h,w)
  '''
  def __init__(self, img, h,w, is_ffe=False):
      super().__init__()
      img = img      
      transform = Compose([
      Resize((h,w)),
      ToTensor(),
      ])
      # print("img : ", img.size)
      # img = transform(img)
      img = torch.from_numpy(img).unsqueeze(0)
      if(len(img.shape)==4) :
        self.pixels = img.permute(1, 2, 3, 0).view(-1, 1)
        print("pixels shape = " , self.pixels.shape)
      else :
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        print("self.pixels : ", self.pixels.shape)
      
      if(is_ffe):
        self.coords = self.get_mgrid2(h*2, w*2, 2) 
      else:
        self.coords = self.get_mgrid(h*2, w*2, 2)
      
  def __len__(self):
      return 1

  def __getitem__(self, idx):    
      if idx > 0: raise IndexError
          
      return self.coords, self.pixels

  def get_mgrid(self, h,w, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    h,w = Height, Width
    dim: int'''
    x_coords = torch.linspace(-1, 1, w)
    y_coords = torch.linspace(-1, 1, h)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='xy')
    mgrid = torch.stack((x_grid.flatten(), y_grid.flatten()), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

  def get_mgrid2(self, h,w, dim=2):
    #Grid Generation for FFE
    '''Generates a grid of (x,y,...) coordinates in a range of -1 to 1.
    h,w = Height, Width
    dim: int'''
    x_coords = torch.linspace(-1, 1, w)
    y_coords = torch.linspace(-1, 1, h)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='xy')
    mgrid = torch.stack((x_grid, y_grid), dim=-1)
    mapping_size = 128
    num_channels = 2
    
    x = GaussianFourierFeatureTransform(num_channels, mapping_size, 5)(mgrid.permute(2, 0,1).unsqueeze(0))
    
    x = x.reshape(num_channels*mapping_size, w*h)
    x = x.permute(1, 0)
    return x
  


#Testing
'''
orig_img = Image.fromarray(lf_gt)
img_prep = ImagePreparation(orig_img, 87, 96, is_ffe=True)
print(img_prep.pixels.shape,img_prep.coords.shape)
'''