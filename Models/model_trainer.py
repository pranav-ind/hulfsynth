import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np



from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from datetime import datetime

import lightning as pl
from lightning.pytorch.loggers import WandbLogger
import wandb


import monai
from monai.losses import DiceLoss, DiceCELoss, SoftclDiceLoss, DiceFocalLoss, NACLLoss
from kornia.losses import total_variation

from Models.models import Siren, Finer
from Models.model import MLP, initialize_siren_weights, SineLayer, ReLULayer
from Utils.utils import psnr, get_device, norm, normalize_psnr



class ContrastModulation:
  #2D
  def __init__(self, ):
    self.hf_chunk_size = (96, 96, 4)
    self.chunk_size_lf = (96//2, 96//2, 4)
    self.hf_size = (172, 192, 192)
    self.lf_size = (172//2, 192//2, 192)

  
  def forward(self, output_image, output_seg, M):
    hf_chunk_size = self.hf_chunk_size #= config["size"]
      
    #Weighted Sum of Segmentation Probabilities and Image Intensities

    imgs_list = [F.interpolate((output_seg[:,i].reshape(hf_chunk_size) * output_image.reshape(hf_chunk_size)).permute(2,0,1).unsqueeze(0), scale_factor=0.5).squeeze(0).permute(1,2,0) for i in range(output_seg.shape[-1])]
    wm_img = imgs_list[0] * M[0]
    gm_img = imgs_list[1] * M[1]
    csf_img = imgs_list[2] * M[2]
    bg_img = imgs_list[3]


    # Recombination of downsampled tissues 
    lf_img = csf_img + gm_img + wm_img #+ bg_img
    return lf_img.flatten().unsqueeze(0)


class ModelTrainerModule(pl.LightningModule):
    def __init__(self,
                 network: MLP,
                 hf_gt_im: torch.Tensor,
                 lf_gt_im: torch.Tensor, 
                 lf_gt_seg: torch.Tensor,
                 config: dict,
                 lr: float = 0.001,
                 name: str = "",
                 eval_interval: int = 100,
                 visualization_intervals: List[int] = [0, 100, 500, 1000, 5000, 10000],
                ):
        super().__init__()
        self.lr = lr
        self.network = network

        # Logging
        self.name = name
        self.hf_gt_im = hf_gt_im
        self.lf_gt_im = lf_gt_im
        self.lf_gt_seg = lf_gt_seg
        self.eval_interval = eval_interval
        self.visualization_intervals = visualization_intervals
        self.progress_ims = []
        self.scores = []
        self.dice = DiceCELoss(include_background=True,squared_pred = True, reduction='mean', jaccard=False)
        self.hf_chunk_size = (96,96,4)
        self.lf_chunk_size = (96//2,96//2,4)
        self.num_classes = 4
        self.config = config
        self.phi = ContrastModulation()
        self.M = [1, 1, 1]

        self.dice_score = monai.metrics.DiceMetric()
        self.dice2 = monai.metrics.GeneralizedDiceScore()
        self.iou_score = monai.metrics.MeanIoU()
        self.psnr_value = monai.metrics.PSNRMetric(max_val = 1.0) #expects shape: BCHWD
        self.ssim_value = monai.metrics.regression.SSIMMetric(spatial_dims=3, data_range=1.0) #expects shape: BCHWD
        self.temp_list1 = []
        self.temp_list2 = []
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def forward(self, coords):
        return self.network(coords)
    
    def compute_loss(self, pred_lf:torch.Tensor, lf_gt:torch.Tensor, output_image_pre:torch.Tensor, pred_seg_dice:torch.Tensor, lf_gt_seg_dice:torch.Tensor, output_seg_pre:torch.Tensor):
        """
        Computes loss for each chunk (batch)
        """
        mse_loss = self.config["l1"] *  F.mse_loss(pred_lf, lf_gt) 
        dice_loss = self.config["l3"] * self.dice(pred_seg_dice, lf_gt_seg_dice)
        tv_loss_img = self.config["l4"] * total_variation(output_image_pre.reshape(self.hf_chunk_size), reduction ='mean').mean()  
        tv_loss_seg = self.config["l5"] * sum([total_variation(output_seg_pre[:,i].reshape(self.hf_chunk_size), reduction='mean').mean() for i in range(output_seg_pre.shape[-1])]) #calculating total variation of each tissue and summing them
        loss = dice_loss + mse_loss + tv_loss_img + tv_loss_seg
        
        print("loss: ", loss.item(), dice_loss.item(), mse_loss.item(), tv_loss_img.item(), tv_loss_seg.item())        
        return loss, mse_loss, dice_loss, tv_loss_img, tv_loss_seg

    def compute_rqs(self, pred_hf_im, pred_hf_seg):
        """
        - Computes Reconstruction Quality Score (rqs): a fusion metric of dice score, IoU score, SSIM and PSNR. 
        - This score is calculated for predicted against observed ULF (not High-Field)
        - This score is used mainly for hyperparameter tuning (and generally indicates "how good" is our reconstruction)
        - Note: this measure is not a final metric that validates our approach. We use the same metrics (dice/iou and ssim/psnr) to measure predicted HF data and observed HF data
        """
        # pred_hf_im, pred_hf_seg  = self.sample_at_resolution(self.hf_gt_im.shape[:-1]) # shape(4 *H_lf, *W_lf, *D)
        # pred_hf_im, pred_hf_seg  = pred_im, pred_seg
        
        #lf_img
        pred_lf_im = (F.interpolate(pred_hf_im.permute(2,0,1).unsqueeze(0), scale_factor=0.5).squeeze(0)).permute(1,2,0)
        pred_lf_im = pred_lf_im.reshape(self.lf_gt_im.shape[:-1])

        #lf_seg
        pred_lf_seg = [(F.interpolate(pred_hf_seg[i].permute(2,0,1).unsqueeze(0), scale_factor=0.5).squeeze(0)).permute(1,2,0).reshape(self.lf_gt_seg.shape[2:]) for i in range(pred_hf_seg.shape[0])] #downsampling pred_seg
        pred_lf_seg = torch.stack(pred_lf_seg,axis = 0) # shape(4 *H_lf, *W_lf, *D)

        #hf_seg
        pred_hf_seg = [pred_hf_seg[i].reshape(self.hf_gt_im.shape[:-1]) for i in range(pred_hf_seg.shape[0])]
        pred_hf_seg = torch.stack(pred_hf_seg,axis = 0).unsqueeze(0) # shape(1,4 *H, *W, *D)
        # print("pred seg: ", pred_hf_seg.shape, pred_lf_seg.shape)

        # RQS = (0.3 * dice) + (0.2 * iou) + (0.3 * ssim) + (0.2 * psnr) -> slightly more biased towards structural indices (dice/ssim)
        psnr_ = self.psnr_value(pred_lf_im.unsqueeze(0).to('cpu'), self.lf_gt_im.permute(3,0,1,2).to('cpu'))
        ssim_ = self.ssim_value(pred_lf_im.unsqueeze(0).unsqueeze(0).to('cpu'), self.lf_gt_im.permute(3,0,1,2).unsqueeze(0).to('cpu'))
        dice_ = self.dice_score(pred_lf_seg.unsqueeze(0).to('cpu'), self.lf_gt_seg.to('cpu'))
        iou_ = self.iou_score(pred_lf_seg.unsqueeze(0).to('cpu'), self.lf_gt_seg.to('cpu'))

        dice2 = self.dice2(pred_lf_seg.unsqueeze(0).to('cpu'), self.lf_gt_seg.to('cpu'))
        print("dice_lf: ", dice_, "iou_lf: ", iou_, "dice2: ", dice2)
        dice_ = dice_.mean()
        iou_ = iou_.mean()
        dice2 = dice2.mean()
        normalized_psnr_ = normalize_psnr(psnr_) #normalizing to [0, 1] i.e., maps : [15.0, 30.0] -> [0, 1]
        
        seg_score = (0.3 * dice2) + (0.2 * iou_) #calculates how the structural fidelity of ULF segmentations
        img_score = (0.3 * ssim_) + (0.2 * normalized_psnr_) #calculates image metrics of ULF predictions
        rqs_ = seg_score + img_score

        # return rqs_, dice_, iou_, ssim_, psnr_, normalized_psnr_
        return rqs_, dice2, iou_, ssim_, psnr_, normalized_psnr_





    def training_step(self, batch, batch_idx):
        coords, lf_batch, lf_batch_seg = batch
        coords = coords.view(-1, coords.shape[-1]) #coord input of each batch
        lf_batch = lf_batch.view(-1, lf_batch.shape[-1]) #lf_gt of each batch
        # print("gt_batch shapes: ",coords.shape, lf_batch.shape, lf_batch_seg.shape)
        
        output_image, output_image_pre, output_seg, output_seg_pre = self.forward(coords)
        output_image_lf = F.interpolate(output_image.unsqueeze(0).unsqueeze(0).squeeze(-1), scale_factor=0.25).squeeze(0) #TODO: Replace F.interpolate with your forward model
        # output_image_lf = self.phi.forward(output_image, output_seg, self.M)
        
        pred_seg = [(F.interpolate(output_seg[:,i].unsqueeze(0).unsqueeze(0), scale_factor=0.25).squeeze(0).squeeze(0)).reshape(self.lf_chunk_size) for i in range(output_seg.shape[-1])] #downsampling pred_seg
        pred_seg = torch.stack(pred_seg,axis = 0).unsqueeze(0) # shape(1,4 48, 48, 4)
        lf_batch_seg = [lf_batch_seg[:,i].reshape(48,48,4) for i in range(lf_batch_seg[0].shape[0])]
        lf_batch_seg = torch.stack(lf_batch_seg,axis = 0).unsqueeze(0) # shape(1,4 48, 48, 4)
        
        # print('Outputs: ', output_image.shape, output_seg.shape, output_image_lf.shape, lf_batch_seg.shape)

        #Compute Losses
        loss, mse_loss, dice_loss, tv_loss_img, tv_loss_seg = self.compute_loss(output_image_lf, lf_batch, output_image_pre, pred_seg, lf_batch_seg, output_seg_pre)


        
        #HF metrics over training 
        if (self.current_epoch % 50 )== 0: #logging every epoch is expensive ; therefore logging in intervals of 50
            pred_im, pred_seg  = self.sample_at_resolution(self.hf_gt_im.shape[:-1]) #TODO: move HF validation metrics to another method
            psnr_hf =  self.psnr_value(pred_im.unsqueeze(0).unsqueeze(0).to('cpu'), self.hf_gt_im.to(pred_im.device).permute(3,0,1,2).unsqueeze(0).to('cpu'))
            ssim_hf =  self.ssim_value(pred_im.unsqueeze(0).unsqueeze(0).to('cpu'), self.hf_gt_im.to(pred_im.device).permute(3,0,1,2).unsqueeze(0).to('cpu'))

            print("psnr_hf: ", psnr_hf, "ssim_hf: ", ssim_hf)
            
            
            self.temp_list1.append(pred_im)
            self.temp_list2.append(pred_seg)
            # print('diff: ', (self.temp_list1[0] - pred_im).mean(), (self.temp_list2[0] - pred_seg).mean())
            
            #RQS: LF metrics over training
            rqs_, dice_, iou_, ssim_, psnr_, normalized_psnr_ = self.compute_rqs(pred_im, pred_seg)
            
            final_img = (pred_im * pred_seg[1]) + (pred_im * pred_seg[2]) + (pred_im * pred_seg[3]) 
            
            wandb.log({ 
            "psnr_hf": psnr_hf.item(), "ssim_hf": ssim_hf.item(),
            "psnr_lf": psnr_.item(), "normalized_psnr_lf": normalized_psnr_.item(), "ssim_lf": ssim_.item(), 
            "dice_lf": dice_.item(), "iou_lf": iou_.item(), "RQS": rqs_.item(),
            "total_loss": loss.item(), "mse": mse_loss.item(), "seg": dice_loss.item(), "tv_seg": tv_loss_seg.item(), "tv_img": tv_loss_img.item(), 
            "pred_img": wandb.Image(norm(pred_im[:,:,95].unsqueeze(0)), mode='L'), "pred2_seg": wandb.Image(pred_seg[2,:,:,95].unsqueeze(0), mode='L'), "pred1_seg": wandb.Image(pred_seg[1,:,:,95].unsqueeze(0), mode='L'), "pred3_seg": wandb.Image(pred_seg[3,:,:,95].unsqueeze(0), mode='L'), #adding channel dimension with unsqueeze(0)
            "final_img": wandb.Image(norm(final_img[:,:,95].unsqueeze(0)), mode='L'),
            # "pred_img": wandb.Image(norm(pred_im[:,:,95]), mode='L'), "pred2_seg": wandb.Image(pred_seg[2,:,:,95].unsqueeze(0), mode='L'), "pred1_seg": wandb.Image(pred_seg[1,:,:,95].unsqueeze(0), mode='L'), "pred3_seg": wandb.Image(pred_seg[3,:,:,95].unsqueeze(0), mode='L') #adding channel dimension with unsqueeze(0)
            }) 
        # wandb_logger.log_image(key="pred", images=[norm(pred_im[:,:,90]).unsqueeze(0), norm(pred_im[:,:,95]).unsqueeze(0), pred_seg[2,:,:,95].unsqueeze(0)], caption=["slice: 90", "slice: 95", "seg_2_slice: 95"]) #adding channel dimension with unsqueeze(0)
        return loss


    @torch.no_grad()
    def sample_at_resolution(self, resolution: Tuple[int, ...]):
        print("sampling at resolution: ", resolution)
        """ Evaluate our INR on a grid of coordinates in order to obtain an image. """
        meshgrid = torch.meshgrid([torch.arange(0, i, device=self.device) for i in resolution], indexing='ij')
        coords = torch.stack(meshgrid, dim=-1)
        coords_norm = coords / torch.tensor(resolution, device=self.device) * 2 - 1
        coords_norm_ = coords_norm.reshape(-1, coords.shape[-1])
        predictions_, _, pred_seg_, _ = self.forward(coords_norm_)
        predictions = predictions_.reshape(resolution)
        resolution_seg = list(resolution) + [pred_seg_.shape[-1]] #adding num_tissues to the resolution shape
        pred_seg_ = pred_seg_.reshape(resolution_seg)
        pred_seg = [pred_seg_[:,:,:,i].reshape(resolution) for i in range(pred_seg_.shape[-1])]
        pred_seg = torch.stack(pred_seg, axis = 0)
        #output shapes- predictions: (H, W, D,); pred_seg: (num_classes, H, W, D)
        return predictions, pred_seg