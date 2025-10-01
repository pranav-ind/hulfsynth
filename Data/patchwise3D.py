

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np
from Utils.utils import get_full_img, norm, get_device, dice_stack_helper, get_model, ClearCache


class RandomPointsDataset(Dataset):
    """
    Samples random voxel indices from the LF image, retrieves voxel values and segmentation labels, downsamples them, and returns the normalized coordinates and voxel values.
    Adopted from: https://github.com/INR4MICCAI/INRTutorial/tree/main
    """
    def __init__(self, hf_image: torch.Tensor, lf_image:torch.Tensor, lf_gt_seg_dice:torch.Tensor, points_num: int = 96*96*4, downsampled_points: int = 48*48*4):

        super().__init__()
        self.device = get_device()
        self.points_num = points_num
        assert hf_image.dtype == torch.float32
        assert lf_image.dtype == torch.float32
        self.hf_image = hf_image.to(self.device)  # (H, W, ..., C)
        self.lf_image = lf_image.to(self.device)  # (H, W, ..., C)
        self.lf_gt_seg_dice = lf_gt_seg_dice.permute(1,2,3,4,0).to(self.device) #(tissues, H,W,D, C)
        self.dim_sizes = self.lf_image.shape[:-1]  # Size of each spatial dimension
        self.downsampled_points = downsampled_points
        self.coord_size = len(self.hf_image.shape[:-1])  # Number of spatial dimensions
        # self.value_size = self.lf_image.shape[-1]  # Channel size

    def __len__(self):
        return 1


    def __getitem__(self, idx: int):
        """
        This method performs core functionality of sampling random spatial coordinates from the volume, retrieving corresponding voxel intensity values and segmentation masks,
        downsampling the data to match ULF resolution, and normalizing coordinates.
        
        downsampling: scale_factor = 0.25 (H and W each by a factor of 0.5)

        Returns:
            tuple: A tuple containing:
                - point_coords_norm (torch.Tensor): Normalized coordinates in range [-1, 1] with shape (B, *points_num, coord_size) #e.g. : (1, 36864, 3)
                - voxel_values (torch.Tensor): Downsampled intensity values at sampled points with shape (B, *downsampled_points)  #e.g. : (1, 9216)
                - voxel_values_seg (torch.Tensor): Downsampled segmentation values for all tissues with shape (B, num_tissues, *downsampled_points) #e.g. : (1, 4, 9216)
        """                                        
        
        
        
        
        point_indices = [torch.randint(0, i, (self.points_num,), device=self.device) for i in self.dim_sizes] # Create random sample of voxel indices
        # point_indices = [torch.randint(0, i, (self.points_num,),) for i in self.dim_sizes] # Create random sample of voxel indices
    
        #Retrival
        voxel_values = self.lf_image[tuple(point_indices)] # Retrieve image voxel values from selected indices
        voxel_values_seg = [self.lf_gt_seg_dice[i][tuple(point_indices)] for i in range(self.lf_gt_seg_dice.shape[0])] #Retrieve image values from selected indices for each tissue
        voxel_values_seg = torch.stack(voxel_values_seg,axis = 0) #list to stack
        
        #Downsampling to match ULF resolution
        # voxel_values = F.interpolate(voxel_values.unsqueeze(0).unsqueeze(0).squeeze(-1), scale_factor=0.25).squeeze(0).squeeze(0) #downsampling lf_gt
        voxel_values = F.interpolate(voxel_values.unsqueeze(0).unsqueeze(0).squeeze(-1), size=self.downsampled_points).squeeze(0).squeeze(0) #downsampling lf_gt
        print(voxel_values_seg.shape)
        # voxel_values_seg = [F.interpolate(voxel_values_seg[i].unsqueeze(0).unsqueeze(0).squeeze(-1), scale_factor=0.25).squeeze(0).squeeze(0) for i in range(self.lf_gt_seg_dice.shape[0])] #downsampling lf_gt_seg
        voxel_values_seg = [F.interpolate(voxel_values_seg[i].unsqueeze(0).unsqueeze(0).squeeze(-1), size=self.downsampled_points).squeeze(0).squeeze(0) for i in range(self.lf_gt_seg_dice.shape[0])] #downsampling lf_gt_seg
        voxel_values_seg = torch.stack(voxel_values_seg,axis = 0)
        print(voxel_values_seg.shape)

        #Normalizing coords
        point_coords = torch.stack(point_indices, dim=-1) # Convert point indices into normalized [-1.0, 1.0] coordinates
        spatial_dims = torch.tensor(self.dim_sizes, device=self.device)
        # spatial_dims = torch.tensor(self.dim_sizes)
        point_coords_norm = point_coords / (spatial_dims / 2) - 1

        return point_coords_norm, voxel_values, voxel_values_seg



#ToDo: Write a unit test case for this class