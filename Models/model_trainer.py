import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np



from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import math
import pprint


import lightning as pl
from lightning.pytorch.loggers import WandbLogger
import wandb


import monai
from monai.losses import DiceLoss, DiceCELoss, SoftclDiceLoss, DiceFocalLoss, NACLLoss #, PerceptualLoss
from kornia.losses import total_variation
from deepinv.utils.demo import load_example
from deepinv.loss.metric import LPIPS, HaarPSI



from Models.model import MLP, initialize_siren_weights, SineLayer, ReLULayer
from Utils.utils import psnr, get_device, norm, normalize_psnr, MSLC, piqe_score
from Utils.plotting_utils import plot_2_images, plot_4_images
from Utils.plotting_utils2 import plot_8_images_2rows



class ContrastModulation:
    def __init__(self, config):
        self.hf_chunk_size = config["hf_chunk_size"] #(96, 96, 4)
        self.lf_chunk_size = config["lf_chunk_size"] #(96//2, 96//2, 4)
        self.hf_size = config["size"] #(172, 192, 192)
        self.lf_size = config["size_lf"] #(172//2, 192//2, 192)
        self.points_num = config["points_num"] #96*96*4
        self.downsampled_points = config["downsampled_points"] #48*48*4
        self.config = config

    def add_rician(self, size_lf, v=1e-1, s=1e-1):
        
        '''
            Adding Rician Noise using the magnitude of a Bivariate Normal Distribution with non-zero mean
            Reference : https://en.wikipedia.org/wiki/Rice_distribution
        '''
        N = 1
        for i in (size_lf):
            N = N * i  #Num of samples
        noise = np.random.normal(scale=s, size=(N, 2)) + [[v,0]]
        noise = np.linalg.norm(noise, axis=1)
        return (noise.reshape(size_lf))
  
    def forward(self, output_image, output_seg, M):
        hf_chunk_size = self.hf_chunk_size


        #Generating tissue-type intensities : downsample(seg*img) 
        imgs_list = [(F.interpolate((output_seg[:,i].reshape(hf_chunk_size) * output_image.reshape(hf_chunk_size)).flatten().unsqueeze(0).unsqueeze(0), size=self.downsampled_points).squeeze(0)) for i in range(output_seg.shape[-1])]
        
        #Weighted Sum of Segmentation Probabilities and Image Intensities
        bg_img = (imgs_list[0]).reshape(self.lf_chunk_size)
        wm_img = (imgs_list[1]).reshape(self.lf_chunk_size) * M[0]
        gm_img = (imgs_list[2]).reshape(self.lf_chunk_size) * M[1]
        csf_img = (imgs_list[3]).reshape(self.lf_chunk_size) * M[2] 
        

        del imgs_list
        lf_img = csf_img + gm_img + wm_img #+ bg_img #shape (1, *lf_chunk)  # Recombination of downsampled tissues 
        
        rician_noise = torch.from_numpy(self.add_rician(lf_img.shape)).to(lf_img.device) #adding rician noise
        mask = torch.where(lf_img>0 ,1.0, 0.0)
        lf_img += (rician_noise * mask) #adding noise only to foreground voxels
        return lf_img.flatten().unsqueeze(0)

    def forward_inference(self, pred_seg, pred_img):
            
        #forward function to be used at inference
        downsampled_points = self.config["downsampled_points"]
        M = self.config["M"]
        
        imgs_list = [(F.interpolate((pred_seg[i] * pred_img).permute(2,0,1).unsqueeze(0), size=self.lf_size[:-1]).squeeze(0).permute(1,2,0)) for i in range(pred_seg.shape[0])]
        # imgs_list = [(F.interpolate((pred_seg[i].reshape(self.hf_size) * pred_img.reshape(self.lf_size)).flatten().unsqueeze(0).unsqueeze(0), scale=0.5).squeeze(0)) for i in range(pred_seg.shape[0])]
        bg_img = (imgs_list[0]).reshape(self.lf_size)
        wm_img = (imgs_list[1]).reshape(self.lf_size) * M[0]
        gm_img = (imgs_list[2]).reshape(self.lf_size) * M[1]
        csf_img = (imgs_list[3]).reshape(self.lf_size) * M[2]


        lf_img = wm_img + gm_img + csf_img #+ bg_img
        rician_noise = torch.from_numpy(self.add_rician(lf_img.shape)).to(lf_img.device) #adding rician noise
        mask = torch.where(lf_img>0 ,1.0, 0.0)
        lf_img += (rician_noise * mask) #adding noise only to foreground voxels
        return lf_img




class ModelTrainerModule(pl.LightningModule):
    def __init__(self,
                wandb_logger: WandbLogger,
                 network: MLP,
                 hf_gt_im: torch.Tensor,
                 lf_gt_im: torch.Tensor, 
                 lf_gt_seg: torch.Tensor,
                 hf_gt_seg: torch.Tensor,
                 config: dict,
                 lr: float = 0.001,
                 name: str = "",
                 eval_interval: int = 100,
                 visualization_intervals: List[int] = [0, 100, 500, 1000, 5000, 10000],

                ):
        super().__init__()
        self.lr = lr
        self.network = network
        self.config = config
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Logging
        self.wandb_logger = wandb_logger
        self.name = name 
        self.hf_gt_im = hf_gt_im.to(self.dev) #(H, W, D, 1)
        self.lf_gt_im = lf_gt_im.to(self.dev)  #(H_lf, W_lf, D_lf, 1)
        self.lf_gt_seg = lf_gt_seg.to(self.dev) #[1, 4, H_lf, W_lf, D]
        self.hf_gt_seg = hf_gt_seg.to(self.dev) #[1, 4, H, W, D]
        self.eval_interval = eval_interval
        self.visualization_intervals = visualization_intervals
        self.progress_ims = []
        self.scores = []
        self.dice = DiceCELoss(include_background=True, squared_pred = True, reduction='mean', jaccard=False)
        self.ssim_loss = monai.losses.ssim_loss.SSIMLoss(spatial_dims=3, win_size = (11,11,4))
        self.hf_chunk_size = config["hf_chunk_size"] #(96,96,4)
        self.lf_chunk_size = config["lf_chunk_size"] #(96//2,96//2,4)
        self.points_num = config["points_num"] #96*96*4
        self.downsampled_points = config["downsampled_points"] #48*48*4
        self.max_epochs = config["epochs"]
        self.num_classes = 4
        
        self.phi = ContrastModulation(self.config)
        self.M = config["M"] #[0.75, 0.9, 0.9]

        self.dice_score = monai.metrics.DiceMetric()
        self.dice2 = monai.metrics.GeneralizedDiceScore()
        self.iou_score = monai.metrics.MeanIoU()
        self.psnr_value = monai.metrics.PSNRMetric(max_val = 1.0) #expects shape: BCHWD
        self.ssim_value = monai.metrics.regression.SSIMMetric(spatial_dims=3, data_range=1.0) #expects shape: BCHWD
        self.lpips = LPIPS()
        self.haar = HaarPSI(norm_inputs="clip")
        self.mslc = MSLC()
        # self.perceptual_loss = PerceptualLoss(spatial_dims=3,network_type='medicalnet_resnet50_23datasets', is_fake_3d=False)
        self.gradient_loss = GradientLoss()
        pprint.pprint(self.config)
        
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=0.0) #weight_decay acts as L2 regularizor; increase it's weighting if your gradients are not smooth/stable.

    def forward(self, coords):
        return self.network(coords)
        

    
    def compute_loss(self, pred_lf:torch.Tensor, lf_gt:torch.Tensor, output_image_pre:torch.Tensor, pred_seg_dice:torch.Tensor, lf_gt_seg_dice:torch.Tensor, output_seg_pre:torch.Tensor):
        """
        Computes loss for each chunk (batch)
        """
     
        
        mse_loss = (self.config["l1"]) * (0.1 * F.l1_loss(pred_lf, lf_gt))# + 0.25 * F.mse_loss(pred_lf, lf_gt)) #L1 + L2 combinational loss
        #grad_loss = self.config["l1"] * (self.gradient_loss(pred_lf.reshape(self.config["lf_chunk_size"]).unsqueeze(0),lf_gt.reshape(self.config["lf_chunk_size"]).unsqueeze(0)))

        dice_loss = self.config["l3"] * self.dice(pred_seg_dice, lf_gt_seg_dice)
       
        tv_loss_img = (self.config["l4"][0] * (total_variation(output_image_pre.reshape(self.hf_chunk_size)* output_seg_pre[:,0].reshape(self.hf_chunk_size), reduction ='mean').mean())) + (self.config["l4"][1] * (total_variation(output_image_pre.reshape(self.hf_chunk_size)* output_seg_pre[:,1].reshape(self.hf_chunk_size), reduction ='mean').mean())) + (self.config["l4"][2] * (total_variation(output_image_pre.reshape(self.hf_chunk_size)* output_seg_pre[:,2].reshape(self.hf_chunk_size), reduction ='mean').mean())) + (self.config["l4"][3] * (total_variation(output_image_pre.reshape(self.hf_chunk_size)* output_seg_pre[:,3].reshape(self.hf_chunk_size), reduction ='mean').mean()))
        tv_loss_seg = self.config["l5"][0] * (total_variation(output_seg_pre[:,0].reshape(self.hf_chunk_size), reduction='mean').mean()) + (self.config["l5"][1] * (total_variation(output_seg_pre[:,1].reshape(self.hf_chunk_size), reduction='mean').mean())) +  (self.config["l5"][2] * (total_variation(output_seg_pre[:,2].reshape(self.hf_chunk_size), reduction='mean').mean())) + (self.config["l5"][3] * (total_variation(output_seg_pre[:,3].reshape(self.hf_chunk_size), reduction='mean').mean()))        
        loss = dice_loss + mse_loss + tv_loss_img + tv_loss_seg #+ grad_loss
        print("loss: ", loss.item(), mse_loss.item(),  dice_loss.item(),  tv_loss_img.item(), tv_loss_seg.item())
        
        return loss, mse_loss, dice_loss, tv_loss_img, tv_loss_seg

    def compute_rqs(self, pred_hf_im, pred_hf_seg, pred_lf_im, pred_lf_seg):
        # pred_hf_im, pred_hf_seg were used in earlier version for validation
        """
        - Computes Reconstruction Quality Score (rqs): a fusion metric of dice score, IoU score, SSIM and PSNR. 
        - This score is calculated for predicted against observed ULF (not High-Field)
        - This score is used for hyperparameter tuning to indicate the quality of our reconstruction
        - Note: this measure is not a final metric that validates our approach. We use the same metrics (dice/iou and ssim/psnr) to measure predicted HF data and observed HF data
        """
        
        slice_num = self.config["slice"]
        pred_lf_im = pred_lf_im.reshape(self.lf_gt_im.shape[:-1])

        
        psnr_ = self.psnr_value(pred_lf_im.unsqueeze(0), self.lf_gt_im.permute(3,0,1,2))
        ssim_ = self.ssim_value(pred_lf_im.unsqueeze(0).unsqueeze(0), self.lf_gt_im.permute(3,0,1,2).unsqueeze(0))
        dice_ = self.dice_score(pred_lf_seg.unsqueeze(0), self.lf_gt_seg).mean()
        iou_ = self.iou_score(pred_lf_seg.unsqueeze(0), self.lf_gt_seg).mean()
        dice2 = self.dice2(pred_lf_seg.unsqueeze(0), self.lf_gt_seg).mean()

        middle_slices = 3
        lpips_ = self.lpips(pred_lf_im.unsqueeze(0).permute(3,0,1,2)[slice_num-middle_slices:slice_num+middle_slices], self.lf_gt_im[:,:,:,0].unsqueeze(0).permute(3,0,1,2)[slice_num-middle_slices:slice_num+middle_slices]).mean() #computing score for middle 10 slices
        haar_ = self.haar(pred_lf_im.unsqueeze(0).permute(3,0,1,2)[slice_num-middle_slices:slice_num+middle_slices], self.lf_gt_im[:,:,:,0].unsqueeze(0).permute(3,0,1,2)[slice_num-middle_slices:slice_num+middle_slices]).mean()
        mslc_ = torch.tensor([self.mslc(pred_lf_im[:,:,i].cpu().numpy()) for i in range(slice_num-middle_slices, slice_num+middle_slices)]).mean() #computing MSLC scor for middle 10 slices
        piqe_ = torch.tensor([piqe_score(pred_lf_im[:,:,i].cpu().numpy()) for i in range(slice_num-middle_slices, slice_num+middle_slices)]) #computing PIQE scor for middle 10 slices
        piqe_ = piqe_.mean()
        normalized_psnr_ = normalize_psnr(psnr_) #normalizing to [0, 1] i.e., maps : [15.0, 30.0] -> [0, 1]
        mslc_inv = 1.0 - mslc_

        seg_score = (0.3 * dice2) + (0.2 * iou_) #calculates how the structural fidelity of ULF segmentations
        img_score = (0.35 * ssim_) + (0.15 * mslc_inv)
        rqs_ = seg_score + img_score #(35% SSIM + 15% MSLC + 50% segmentaion)
    
        return rqs_, dice2, iou_, ssim_, psnr_, normalized_psnr_, lpips_, haar_, mslc_, piqe_





    def training_step(self, batch, batch_idx):
        
        slice_num = self.config["slice"]
        lf_chunk_size = self.lf_chunk_size

        coords, lf_batch, lf_batch_seg = batch
        coords = coords.view(-1, coords.shape[-1]) #coord input of each batch
        lf_batch = lf_batch.view(-1, lf_batch.shape[-1]) #lf_gt of each batch
        
        output_image, output_image_pre, output_seg, output_seg_pre = self.forward(coords) #shape(*hf_chunk), (*hf_chunk), (*hf_chunk, 4), (*hf_chunk, 4)
        output_image_lf = self.phi.forward(output_image, output_seg, self.M)
        
        
        pred_seg = [(F.interpolate(output_seg[:,i].unsqueeze(0).unsqueeze(0), size=self.downsampled_points).squeeze(0).squeeze(0)).reshape(self.lf_chunk_size) for i in range(output_seg.shape[-1])] #downsampling pred_seg
        pred_seg = torch.stack(pred_seg,axis = 0).unsqueeze(0) # shape(1,4 48, 48, 4)
        lf_batch_seg = [lf_batch_seg[:,i].reshape(lf_chunk_size) for i in range(lf_batch_seg[0].shape[0])]
        lf_batch_seg = torch.stack(lf_batch_seg,axis = 0).unsqueeze(0) # shape(1,4 48, 48, 4)
    

        #Compute Losses
        loss, mse_loss, dice_loss, tv_loss_img, tv_loss_seg = self.compute_loss(output_image_lf, lf_batch, output_image_pre, pred_seg, lf_batch_seg, output_seg_pre)
        # loss, mse_loss, dice_loss, tv_loss_img, tv_loss_seg = self.compute_loss(output_image_lf, lf_batch, output_image, pred_seg, lf_batch_seg, output_seg_pre) #using output image for regularization

        loss_dict = {"total_loss": loss.item(), "mse": mse_loss.item(), "seg": dice_loss.item(), "tv_seg": tv_loss_seg.item(), "tv_img": tv_loss_img.item()}
        self.wandb_logger.log_metrics(loss_dict, step=self.global_step)
        
        #Image metrics over training 
        if (( (self.current_epoch) % 2500 )== 0): #or (self.current_epoch == (self.max_epochs)-10): #logging every epoch is expensive ; therefore logging in intervals of 2500
            self.inference()

        return loss
    
    def on_train_end(self):
        self.inference()
    
    @torch.no_grad()
    def inference(self):
        """
        Computing all metrices and reporting metrics
        """
        slice_num = self.config["slice"]
        pred_im, pred_hf_seg, pred_lf_im  = self.sample_at_resolution(self.hf_gt_im.shape[:-1]) #
        pred_im = norm(pred_im)
        if(pred_im.min() >= 0) and (pred_im.max() <= 1.0):
            pred_lf_im = pred_lf_im.reshape(self.lf_gt_im.shape[:-1])
            final_img_bg = ((pred_im * pred_hf_seg[1]) + (pred_im * pred_hf_seg[2]) + (pred_im * pred_hf_seg[3]) + (pred_im * pred_hf_seg[0]))  #adding bg 
            final_img = norm((pred_im * pred_hf_seg[1]) + (pred_im * pred_hf_seg[2]) + (pred_im * pred_hf_seg[3])) #+ (pred_im * pred_hf_seg[0])) #no bg

            #lf_seg
            pred_lf_seg = [(F.interpolate(pred_hf_seg[i].permute(2,0,1).unsqueeze(0), size=self.lf_gt_im.shape[:-2]).squeeze(0)).permute(1,2,0).reshape(self.lf_gt_seg.shape[2:]) for i in range(pred_hf_seg.shape[0])] #downsampling pred_seg
            pred_lf_seg = torch.stack(pred_lf_seg,axis = 0) # shape(4 *H_lf, *W_lf, *D)
            rqs_, dice_, iou_, ssim_, psnr_, normalized_psnr_, lpips_, haar_, mslc_, piqe_ = self.compute_rqs(pred_im, pred_hf_seg, pred_lf_im, pred_lf_seg) #Get ULF Image metrics
        

            #hf_seg
            pred_hf_seg = [pred_hf_seg[i].reshape(self.hf_gt_im.shape[:-1]) for i in range(pred_hf_seg.shape[0])]
            pred_hf_seg = torch.stack(pred_hf_seg,axis = 0).unsqueeze(0) # shape(1,4 *H, *W, *D)
            iou_hf = self.iou_score(pred_hf_seg, self.hf_gt_seg).mean()
            # dice2_hf = self.dice2(pred_hf_seg, self.hf_gt_seg).mean() #Generalized dice score
            dice2_hf = self.dice_score(pred_hf_seg, self.hf_gt_seg).mean() #Mean dice score

            
            #Calculating HF Image Metrics
            psnr_hf =  self.psnr_value(pred_im.unsqueeze(0).unsqueeze(0), self.hf_gt_im.permute(3,0,1,2).unsqueeze(0))
            ssim_hf =  self.ssim_value(pred_im.unsqueeze(0).unsqueeze(0), self.hf_gt_im.permute(3,0,1,2).unsqueeze(0))
            lpips_hf = self.lpips(pred_im.unsqueeze(0).permute(3,0,1,2)[slice_num-5:slice_num+5], self.hf_gt_im[:,:,:,0].unsqueeze(0).permute(3,0,1,2)[slice_num-5:slice_num+5]).mean() #computing score for middle 10 slices
            psnr_hf_final =  self.psnr_value(final_img.unsqueeze(0).unsqueeze(0), self.hf_gt_im.permute(3,0,1,2).unsqueeze(0))
            ssim_hf_final =  self.ssim_value(final_img.unsqueeze(0).unsqueeze(0), self.hf_gt_im.permute(3,0,1,2).unsqueeze(0))
            lpips_hf_final = self.lpips(final_img.unsqueeze(0).permute(3,0,1,2)[slice_num-5:slice_num+5], self.hf_gt_im[:,:,:,0].unsqueeze(0).permute(3,0,1,2)[slice_num-5:slice_num+5]).mean() #computing score for middle 10 slices
            haar_hf_final = self.haar(final_img.unsqueeze(0).permute(3,0,1,2)[slice_num-5:slice_num+5], self.hf_gt_im[:,:,:,0].unsqueeze(0).permute(3,0,1,2)[slice_num-5:slice_num+5]).mean()
            mslc_final = torch.tensor([self.mslc(final_img[:,:,i].cpu().numpy()) for i in range(slice_num-5, slice_num+5)]).mean() #computing MSLC scor for middle 10 slices
            piqe_final = torch.tensor([piqe_score(final_img[:,:,i].cpu().numpy()) for i in range(slice_num-5, slice_num+5)]) #computing PIQE scor for middle 10 slices

            print("LF scores: ", rqs_, dice_, iou_, ssim_, psnr_, normalized_psnr_, lpips_, haar_, mslc_, piqe_) #used for hyperparameter tuning
            print("HF scores: ", psnr_hf, ssim_hf, iou_hf, dice2_hf, lpips_hf_final, haar_hf_final, mslc_final, piqe_final)
            piqe_final = piqe_final.mean()
            
            #plots
            print("plotting all images.....")
            titles = ['BG', 'WM', 'GM', 'CSF']
            images_gt = [self.hf_gt_seg[0,0,:,:,slice_num].cpu(), self.hf_gt_seg[0,1,:,:,slice_num].cpu(), self.hf_gt_seg[0,2,:,:,slice_num].cpu(), self.hf_gt_seg[0,3,:,:,slice_num].cpu()]
            images_pred = [pred_hf_seg[0,0,:,:,slice_num].cpu(), pred_hf_seg[0, 1,:,:,slice_num].cpu(), pred_hf_seg[0, 2,:,:,slice_num].cpu(), pred_hf_seg[0, 3,:,:,slice_num].cpu()]
            fig_hf_seg, _ = plot_8_images_2rows(images_gt + images_pred, titles=titles, figsize=(12, 6), cmap='gray', suptitle='HF Segmentations ', row_labels=['observed' +' (slice: ' + str(slice_num) +')', 'predicted '+ '(slice: ' + str(slice_num) +')'], save_path=None, dpi=100)        
            images_gt = [self.lf_gt_seg[0,0,:,:,slice_num].cpu(), self.lf_gt_seg[0,1,:,:,slice_num].cpu(), self.lf_gt_seg[0,2,:,:,slice_num].cpu(), self.lf_gt_seg[0,3,:,:,slice_num].cpu()]
            images_pred = [pred_lf_seg[0,:,:,slice_num].cpu(), pred_lf_seg[ 1,:,:,slice_num].cpu(), pred_lf_seg[2,:,:,slice_num].cpu(), pred_lf_seg[3,:,:,slice_num].cpu()]
            fig_lf_seg, _ = plot_8_images_2rows(images_gt + images_pred, titles=titles, figsize=(12, 6), cmap='gray', suptitle='ULF Segmentations ', row_labels=['observed' +' (slice: ' + str(slice_num) +')', 'predicted '+ '(slice: ' + str(slice_num) +')'], save_path=None, dpi=100)
            fig_hf_im = plot_4_images(self.hf_gt_im[:,:,slice_num,0].cpu(), pred_im[:,:,slice_num].cpu(), final_img[:,:,slice_num].cpu(), final_img_bg[:,:,slice_num].cpu(), title1="hf_obs", title2="pred_hf", title3="final_hf", title4="final_hf_bg")
            fig_lf_im = plot_2_images(self.lf_gt_im[:,:,slice_num,0].cpu(), pred_lf_im[:,:,slice_num].cpu(), title1="ulf_obs", title2="pred_ulf")
            
            print("Logging image metrics in wandb..... ")
            #wandb logging
            log_dict = { 
            "psnr_hf": psnr_hf.item(), "ssim_hf": ssim_hf.item(), 
            "psnr_hf_final": psnr_hf_final.item(), "ssim_hf_final": ssim_hf_final.item(),
            "dice_hf": dice2_hf.item(), "iou_hf": iou_hf.item(),
            "lpips_hf": lpips_hf.item(), 
            "lpips_hf_final": lpips_hf_final.item(), "haar_hf_final": haar_hf_final.item(), "mslc_final": mslc_final.item(), "piqe_final": piqe_final.item(),

            "psnr_lf": psnr_.item(), "normalized_psnr_lf": normalized_psnr_.item(), "ssim_lf": ssim_.item(), 
            "lpips_lf": lpips_.item(), "haar_lf": haar_.item(), "mslc_lf": mslc_.item(), "piqe_lf": piqe_.item(),
            "dice_lf": dice_.item(), "iou_lf": iou_.item(), "RQS": rqs_.item(),
            'fig_hf' : wandb.Image(fig_hf_im), 'fig_ulf' : wandb.Image(fig_lf_im), 'fig_hf_seg' : wandb.Image(fig_hf_seg), 'fig_uf_seg' : wandb.Image(fig_lf_seg)
            }
            self.wandb_logger.log_metrics(log_dict)
        else:
            print("Skipping computing scores as the range of pred_im is not in [0, 1]")
    

    @torch.no_grad()
    def get_coords(self,resolution: Tuple[int, ...]):
        print("sampling at resolution: ", resolution)
        device = self.device if (self.device!= None) else self.dev
        meshgrid = torch.meshgrid([torch.arange(0, i, device=device) for i in resolution], indexing='ij')
        coords = torch.stack(meshgrid, dim=-1)
        coords_norm = coords / torch.tensor(resolution, device=device) * 2 - 1
        coords_norm_ = coords_norm.reshape(-1, coords.shape[-1])
        coords_norm_ += (torch.randn_like(coords_norm_) * 0.001) #adding gaussian noise with std = 0.001
        return coords_norm_
    
    @torch.no_grad()
    def sample_at_resolution(self, resolution: Tuple[int, ...]):
        print("sampling at resolution: ", resolution)
        """ Evaluate our INR on a grid of coordinates in order to obtain an image. """
        
        coords_norm_ = self.get_coords(resolution)
        predictions_, _, pred_seg_, _ = self.forward(coords_norm_) 
        resolution_seg = list(resolution) + [pred_seg_.shape[-1]] #adding num_tissues to the resolution shape
        predictions_hf = predictions_.reshape(resolution)
        pred_seg_ = pred_seg_.reshape(resolution_seg)
        pred_seg = [pred_seg_[:,:,:,i].reshape(resolution) for i in range(pred_seg_.shape[-1])]
        pred_seg = torch.stack(pred_seg, axis = 0) 
        print("sample at resolution: ", pred_seg.shape, predictions_hf.shape)
        pred_lf = self.phi.forward_inference(pred_seg, predictions_hf) #contrast modulation
        return predictions_hf, pred_seg, pred_lf #output shapes- predictions_hf: (H, W, D,); pred_seg: (num_classes, H, W, D), pred_lf: (H_lf, W_lf, D_lf)




class PosEncINRLightningModule(ModelTrainerModule):
    """ INR with Fourier Feature positional encoding """
    def __init__(self,
                 pos_encoder: nn.Module,
                 network: MLP,
                 **kwargs):
        super().__init__(network=network, **kwargs)
        self.pos_encoder = pos_encoder

    def configure_optimizers(self):
        # Note that we do NOT optimize the positional encoder's parameters.
        return torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def forward(self, coords):
        # Apply positional encoding, then apply network
        pos_enc_coords = self.pos_encoder(coords)
        print(pos_enc_coords.shape)
        return self.network(pos_enc_coords)





class GradientLoss(nn.Module):
    #edge-prserving loss function used as reconstruction loss
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred, target):
        # Sobel-like gradients
        pred_gx = pred[:, :, :, :-1] - pred[:, :, :, 1:]
        pred_gy = pred[:, :, :-1, :] - pred[:, :, 1:, :]
        
        target_gx = target[:, :, :, :-1] - target[:, :, :, 1:]
        target_gy = target[:, :, :-1, :] - target[:, :, 1:, :]
        
        loss = F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)
        return self.weight * loss