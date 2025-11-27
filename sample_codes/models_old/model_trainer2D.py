import torch.nn as nn

from kornia.losses import total_variation
from monai.losses import DiceLoss, DiceCELoss, SoftclDiceLoss, DiceFocalLoss, NACLLoss
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from monai.losses.dice import *  # NOQA
from IPython.display import clear_output
from monai import metrics
import tempfile, os
import monai


# from utils import SineLayer, gradient, laplace, get_device, GaussianFourierFeatureTransform, get_metrics
# from plotting_utils import plot_2_images, plot_4_images, plot_5_images, loss_plot, plot_results_after_train
# from ImagePreparation import ImagePreparation
# from models import Siren, Finer
from projects.hulfsynth.hulfsynth.sample_codes.ContrastModulation2D import ContrastModulation
from Utils.utils import get_full_img, norm, get_device, dice_stack_helper, get_model


class ModelTrainer(nn.Module):
    def __init__(self, config, lf_dataloader, hf_ground_truth, prior_seg_dice, lf_gt_seg_dice, M):
        super(ModelTrainer, self).__init__()
        self.device = get_device()
        self.M = M
        config["in_features"] = 256 if(config["ffe"]==True) else  2
        print("Features = ", config["in_features"])
        self.config = config
        self.model = get_model(config).to(self.device)
        # self.model = Siren(in_features=config["in_features"], out_features=8, hidden_features=config["hidden_features"], hidden_layers=config["hidden_layers"], outermost_linear=True, first_omega_0 = config["w0"], hidden_omega_0 = config["w0"])
        self.model = self.model.to(self.device)
        self.lr = config["lr"]
        self.scheduler_step_size=config["scheduler_step_size"]
        self.scheduler_gamma = gamma=config["scheduler_gamma"]
        self.size = config["size"]
        self.size_lf = config["size_lf"]
        self.epochs = config["total_steps"]
        self.dice = DiceCELoss(include_background=True,squared_pred = True, reduction='mean', jaccard=False)
        self.psnr = monai.metrics.PSNRMetric(max_val = 1.0)
        self.ssim = monai.metrics.regression.SSIMMetric(spatial_dims=2, data_range=1.0)
        self.phi = ContrastModulation()
        self.model_input, self.ground_truth = next(iter(lf_dataloader))
        self.model_input, self.ground_truth = self.model_input.to(self.device), self.ground_truth.to(self.device)

        self.optim = torch.optim.Adam(lr=self.lr, params=self.model.parameters())#, weight_decay=0.05) #0.05 default
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = self.scheduler_step_size, gamma=self.scheduler_gamma)
        self.hf_ground_truth =  torch.from_numpy(hf_ground_truth).to(torch.float32).to(self.device)
        self.prior_seg_dice = prior_seg_dice.to(torch.float32).to(self.device)
        self.lf_gt_seg_dice = lf_gt_seg_dice.to(torch.float32).to(self.device)
        self.l1, self.l2, self.l3, self.l4, self.l5 = config["l1"], config["l2"], config["l3"], config["l4"], config["l5"]
        self.noise_std = 1

        print("size : ", self.size)


    def train(self):
        
        hf_ground_truth = self.hf_ground_truth
        prior_seg_dice = self.prior_seg_dice
        lf_gt_seg_dice = self.lf_gt_seg_dice

        losses = {"mse" : [], "prior" : [], "seg" : [], "TV_seg" : [], "TV_img" : [] , "Total_Loss" : []} #Dictionary to store losses
        train_metrics = {"psnr" : [], "ssim": []}
        
        hf_gt_normalized = (norm(hf_ground_truth)).unsqueeze(0).unsqueeze(0)
        

        for step in tqdm(range(self.epochs)):
            l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5
            

            # self.model_input += torch.randn_like(self.model_input) * (1.0/32.) *self.noise_std
            model_output_seg_pre, model_output_seg, model_output_img_pre, model_output_img, coords = self.model(self.model_input)
            # print("Outputs shape = ", model_output_seg_pre.shape, model_output_seg.shape, model_output_img.shape)
            lf_output = self.phi.forward(model_output_seg, model_output_img, self.M)



            L_mse = l1 * ((lf_output - self.ground_truth)**2).mean() #MSE Loss
            
            #Prior Loss
            model_output_seg_dice = torch.stack((model_output_seg[:,:,3].reshape(self.size), model_output_seg[:,:,0].reshape(self.size), model_output_seg[:,:,1].reshape(self.size), model_output_seg[:,:,2].reshape(self.size)), dim=0).unsqueeze(0)
            L_prior = l2 * (self.dice(model_output_seg_dice, prior_seg_dice))
            
            #Segmentation Loss
            lf_pred_seg_dice = torch.stack(((model_output_seg[:,:,3].reshape(self.size))[::2, ::2], (model_output_seg[:,:,0].reshape(self.size))[::2, ::2], (model_output_seg[:,:,1].reshape(self.size))[::2, ::2], (model_output_seg[:,:,2].reshape(self.size))[::2, ::2]), dim=0).unsqueeze(0)
            L_seg= l3 * (self.dice(lf_gt_seg_dice, lf_pred_seg_dice))
            
            #Total Variation Loss
            TV_seg = ((l4[0] * total_variation(model_output_seg_pre[0,:,0].view(self.size), reduction='mean')) + (l4[1]* total_variation(model_output_seg_pre[0,:,1].view(self.size), reduction='mean')) + (l4[2] * total_variation(model_output_seg_pre[0,:,2].view(self.size), reduction='mean')) + (l4[3] * total_variation(model_output_seg_pre[0,:,3].view(self.size), reduction='mean')))/4.0
            TV_img = ((l5[0] * total_variation(model_output_img[0,:,0].view(self.size), reduction='mean')) +  (l5[1] * total_variation(model_output_img[0,:,1].view(self.size), reduction='mean')) 
            + (l5[2] * total_variation(model_output_img[0,:,2].view(self.size), reduction='mean')) + (l5[3] * total_variation(model_output_img[0,:,3].view(self.size), reduction='mean')))
            loss = L_mse + L_prior + L_seg  + TV_seg + TV_img #+ grad_reg_seg + grad_reg_img #Total Loss
            
            losses["mse"].append(L_mse.item())
            losses["prior"].append(L_prior.item())
            losses["seg"].append(L_seg.item())
            losses["TV_seg"].append(TV_seg.item())
            losses["TV_img"].append(TV_img.item())
            losses["Total_Loss"].append(loss.item())

            #Metrics
            
            
            hf_int_res_normalized = norm(get_full_img(model_output_seg, model_output_img, self.config)).unsqueeze(0).unsqueeze(0).to(torch.float32)
            # print(hf_int_res_normalized.shape)
            psnr_epoch = self.psnr(hf_gt_normalized, hf_int_res_normalized).item()
            ssim_epoch = self.ssim(hf_gt_normalized, hf_int_res_normalized).item()
            train_metrics['psnr'].append(psnr_epoch)
            train_metrics['ssim'].append(ssim_epoch)
            # torch.cuda.synchronize()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.scheduler.step()
            
            size = self.size
            size_lf = self.size_lf
            # if(step  % 350 == 0): 
            #     clear_output(True)
            #     model_output_seg, model_output_img, hf_int_res_normalized = model_output_seg.detach().cpu(), model_output_img.detach().cpu(), hf_int_res_normalized.detach().cpu()
            #     f = plot_4_images(model_output_img[:,:,0].view(size), model_output_img[:,:,1].view(size), 
            #     model_output_img[:,:,2].view(size), model_output_img[:,:,3].view(size), "WM-Prediction", "GM-Prediction", "CSF-Prediction", "BG-Prediction")
            #     plt.show()
            #     f = plot_4_images(model_output_seg[:,:,0].view(size), model_output_seg[:,:,1].view(size), 
            #     model_output_seg[:,:,2].view(size), model_output_seg[:,:,3].view(size), "WM-Prediction", "GM-Prediction", "CSF-Prediction", "BG-Prediction")
            #     plt.show()
            #     # f = plot_4_images(hf_int_res_normalized[:,:,:,:,0].view(size), hf_int_res_normalized[:,:,:,:,1].view(size), hf_int_res_normalized[:,:,:,:,2].view(size), hf_int_res_normalized[:,:,:,:,3].view(size),"WM-Prediction", "GM-Prediction", "CSF-Prediction", "BG-Prediction")
            #     f = plot_4_images(model_output_img[:,:,0].view(size)*model_output_seg[:,:,0].view(size), model_output_img[:,:,1].view(size)*model_output_seg[:,:,1].view(size), 
            #     model_output_img[:,:,2].view(size)*model_output_seg[:,:,2].view(size), model_output_img[:,:,3].view(size)*model_output_seg[:,:,3].view(size), "WM-Prediction", "GM-Prediction", "CSF-Prediction", "BG-Prediction")
            #     plt.show()
            #     fig_final = plot_final_results_3(config, model_output_seg, model_output_img, hf_ground_truth.detach().cpu(), lf_gt.detach().cpu(), lf_output.detach().cpu())
            #     plt.show()

        return model_output_seg, model_output_img, lf_output, losses, train_metrics


    def inference(self, model ):
        model = model.to(self.device)

        with torch.no_grad():
            model_output_seg_pre, model_output_seg,model_output_img_pre, model_output_img, coords = model.forward(self.model_input)
    
        model_output_seg_pre, model_output_seg, model_output_img, coords = (model_output_seg_pre.to('cpu'), model_output_seg.to('cpu'), model_output_img.to('cpu'), coords.to('cpu'))
        lf_output = self.phi.forward(model_output_seg, model_output_img, self.M)
        
        return model_output_seg, model_output_img, lf_output


