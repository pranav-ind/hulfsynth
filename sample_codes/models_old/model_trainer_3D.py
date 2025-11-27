import torch.nn as nn
import torch.nn.functional as F
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
from Data.load_data_3d import load_data, get_gt_seg
from Utils.utils import get_full_img, norm, get_device, dice_stack_helper, get_model, model_seg_loss, lf_seg_loss, pad_chunk, ClearCache
from Utils.defaults import default_config
from LFSynth.ContrastModulation import ContrastModulation
from tqdm import tqdm
from Utils.defaults import default_config

@torch.no_grad()
def make_grid_3d(D: int, H: int, W: int, device) -> torch.Tensor:
    """
    Returns tensor [D, H, W, 3] with coords in [-1,1].
    """
    zs, ys, xs = (
        torch.linspace(-1, 1, D, device=device),
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
    )
    z, y, x = torch.meshgrid(zs, ys, xs, indexing="ij")
    return torch.stack([x, y, z], dim=-1)  # (D,H,W,3)



class ModelTrainer(nn.Module):
    def __init__(self, config, lf_gt, prior_seg_dice, lf_gt_seg_dice, M):
        super(ModelTrainer, self).__init__()
        self.device = get_device()
        self.M = M
        config["in_features"] = 256 if(config["ffe"]==True) else  3
        print("Features = ", config["in_features"])
        self.config = config
        self.model = get_model(config).to(self.device)
        self.model = torch.compile(self.model) #fastens inference
        self.lr = config["lr"]
        self.scheduler_step_size=config["scheduler_step_size"]
        self.scheduler_gamma = gamma=config["scheduler_gamma"]
        self.size = config["size"]
        self.size_lf = config["size_lf"]
        self.epochs = config["total_steps"]
        
        
        self.chunk = (172*192*192)//32 #init chunk via config
        self.chunk_size = (43*2, 48*2, 48*2) #init chunk_size via config


        self.dice = DiceCELoss(include_background=True,squared_pred = True, reduction='mean', jaccard=False)
        self.psnr = monai.metrics.PSNRMetric(max_val = 1.0)
        self.ssim = monai.metrics.regression.SSIMMetric(spatial_dims=2, data_range=1.0)
        self.phi = ContrastModulation()
        # self.model_input, self.ground_truth = next(iter(lf_dataloader))
        # self.model_input = self.model_input.to(self.device)
        
        self.ground_truth = torch.from_numpy(lf_gt)
        self.ground_truth = self.ground_truth.to(self.device)
        # self.model_input = generate_3d_grid(high_res_shape).to(device)

        self.optim = torch.optim.Adam(lr=self.lr, params=self.model.parameters())#, weight_decay=0.05) #0.05 default
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = self.scheduler_step_size, gamma=self.scheduler_gamma)
        # self.hf_ground_truth =  torch.from_numpy(hf_ground_truth).to(self.device)
        self.prior_seg_dice = prior_seg_dice.to(self.device)
        self.lf_gt_seg_dice = lf_gt_seg_dice.to(self.device)
        self.l1, self.l2, self.l3, self.l4, self.l5 = config["l1"], config["l2"], config["l3"], config["l4"], config["l5"]
        print("init complete")

    def compute_loss(self, pred_lf_chunk, targ_chunk, model_output_seg, targ_seg_chunk, targ_prior_chunk, model_output_seg_pre, model_output_img):

        chunk_size = self.chunk_size
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5
        # print("output shapes : ", lf_output.shape,lf_gt.shape, model_output_img.shape) 
        
        #MSE
        L_mse = l1 * F.mse_loss(pred_lf_chunk, targ_chunk.float()) #/ n_chunks  # normalize
        
        #Segmentation
        lf_pred_seg_dice = lf_seg_loss(model_output_seg, (43, 48, 48))
        targ_seg_chunk = targ_seg_chunk.reshape(lf_pred_seg_dice.shape)
        L_seg= l2 * (self.dice(targ_seg_chunk, lf_pred_seg_dice))

        #Prior
        model_output_seg_dice = torch.stack((model_output_seg[:,:,3].reshape(chunk_size), model_output_seg[:,:,0].reshape(chunk_size), model_output_seg[:,:,1].reshape(chunk_size), model_output_seg[:,:,2].reshape(chunk_size)), dim=0).unsqueeze(0)
        targ_prior_chunk = targ_prior_chunk.reshape(model_output_seg_dice.shape)
        L_prior = l3 * (self.dice(targ_prior_chunk, model_output_seg_dice))

        #Total Variation
        TV_seg = ((l4[0] * total_variation(model_output_seg_pre[0,:,0].view(chunk_size), reduction='mean')) + (l4[1]* total_variation(model_output_seg_pre[0,:,1].view(chunk_size), reduction='mean')) + (l4[2] * total_variation(model_output_seg_pre[0,:,2].view(chunk_size), reduction='mean')) + (l4[3] * total_variation(model_output_seg_pre[0,:,3].view(chunk_size), reduction='mean'))).mean()
        TV_img = ((l5[0] * total_variation(model_output_img[0,:,0].view(chunk_size), reduction='mean')) +  (l5[1] * total_variation(model_output_img[0,:,1].view(chunk_size), reduction='mean')) + (l5[2] * total_variation(model_output_img[0,:,2].view(chunk_size), reduction='mean')) + (l5[3] * total_variation(model_output_img[0,:,3].view(chunk_size), reduction='mean'))).mean()
        
        #Total Loss
        loss = L_mse + L_prior + L_seg  + TV_seg + TV_img #+ grad_reg_seg + grad_reg_img 
        return loss, L_mse, L_prior, L_seg, TV_seg, TV_img

    def train_step(self, losses, model: nn.Module, target_gt: torch.Tensor, target_seg: torch.Tensor, target_prior: torch.Tensor):

        chunk_size = self.chunk_size
        chunk = self.chunk
        grid = self.grid
        # print("grid shape = ", grid.shape, "targets shape = ", targets.shape, "target seg shape = ", target_seg.shape)
        
        N = grid.shape[1]
        n_chunks = (N + chunk - 1) // chunk
        
        loss_per_epoch = 0.0
        mse_per_epoch = 0.0
        seg_per_epoch = 0.0
        prior_per_epoch = 0.0
        tv_seg_per_epoch = 0.0
        tv_img_per_epoch = 0.0
        s2 = 0
        for c in tqdm(range(n_chunks)):                       # micro-batch loop
            s, e = c * chunk, min((c + 1) * chunk, N)
            e2 = s2 + (25344*2) #get this via config (22*24*48)
            # print(s,e,s2,e2)
            
            #HF-chunks
            coord_chunk = grid[:, s:e, :].reshape(-1, 3)    # (B*chunk,3)
            targ_prior_chunk = target_prior[:,:,s:e,:]
            
            #LF-chunks
            targ_chunk  = target_gt[:, s2:e2, :].reshape(-1, 1) # (B*chunk,1)
            targ_seg_chunk  = target_seg[:,:,s2:e2,:] # (1,4,*chunk,1) #lf seg chunk
            
            print("coord chunk, targ chunk", coord_chunk.shape, targ_chunk.shape)
            coord_chunk = coord_chunk.unsqueeze(0)
            targ_chunk = targ_chunk.unsqueeze(0)
            
            model_output_seg_pre, model_output_seg, model_output_img, coords = model(coord_chunk)
            print('model_output_img = ', model_output_img.shape, model_output_seg.shape)
            pred_lf_chunk = self.phi.forward(model_output_seg, model_output_img, self.M) #shape (1,1,*chunk)
            
            loss, L_mse, L_prior, L_seg, TV_seg, TV_img = self.compute_loss(pred_lf_chunk, targ_chunk, model_output_seg, targ_seg_chunk, targ_prior_chunk, model_output_seg_pre, model_output_img) #total_loss per chunk
            
            loss.backward() 

            loss_per_epoch += loss #loss per epoch
            mse_per_epoch += L_mse 
            seg_per_epoch += L_seg
            prior_per_epoch += L_prior
            tv_seg_per_epoch += TV_seg
            tv_img_per_epoch += TV_img
            
            # print(loss_per_epoch)
            
            s2 = e2
        return model, loss_per_epoch, mse_per_epoch, seg_per_epoch, prior_per_epoch, tv_seg_per_epoch, tv_img_per_epoch
    

    def train_inr(self):
        model = self.model.to(self.device).train() #set the model to train mode
        device = self.device
        B, _, D, H, W = 1,1, 192, 172, 192 #set this via config

        
        hf_size = (172, 192, 192)
        lf_size = (88, 96, 192)

        #input 3D-coord grid
        self.grid = make_grid_3d(hf_size[0], hf_size[1], hf_size[2], device)             # (D',H',W',3)
        self.grid = self.grid.view(-1, 3)                                   # (N,3)
        self.grid = self.grid.unsqueeze(0)

        #targets
        lf_gt = self.ground_truth
        lf_gt_seg_dice = self.lf_gt_seg_dice
        prior_seg_dice = self.prior_seg_dice


        #set these shapes via config
        target_gt = F.pad(lf_gt, (0, 0, 0, 0, 0, 1)).to(device).flatten().unsqueeze(-1).unsqueeze(0) #shape : (1, *hf_size,1)
        target_seg = F.pad(lf_gt_seg_dice, (0, 0, 0, 0, 0, 1)).to(device).reshape(1,4, 88*96*192).unsqueeze(-1) #shape : (1, 4, lf_size,1)
        target_prior = prior_seg_dice[:,:,1:-1,:,:].to(device).reshape(1,4, 172*192*192).unsqueeze(-1) #shape : (1, 4, hf_size,1)
        
        #Model settings
        optimizer = self.optim
        
        
        losses = {"mse" : [], "prior" : [], "seg" : [], "TV_seg" : [], "TV_img" : [] , "Total_Loss" : []} #Dictionary to store losses
        
        for ep in (range(self.epochs)):
            optimizer.zero_grad(set_to_none=True)
            model, total_loss, mse_per_epoch, seg_per_epoch, prior_per_epoch, tv_seg_per_epoch, tv_img_per_epoch = self.train_step(losses, model, target_gt, target_seg, target_prior)
            
            # self.total_loss.append(loss_per_epoch) #Total Loss
            losses["mse"].append(mse_per_epoch.item())
            losses["prior"].append(prior_per_epoch.item())
            losses["seg"].append(seg_per_epoch.item())
            losses["TV_seg"].append(tv_seg_per_epoch.item())
            losses["TV_img"].append(tv_img_per_epoch.item())
            losses["Total_Loss"].append(total_loss.item())
            torch.cuda.synchronize()
            optimizer.step()
            # self.scheduler.step()
        
        return model, losses


    def inference(self, model):
        device2 = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        model = model.to(device2)
       
       #TBD : init. grid via config
        grid = make_grid_3d(174, 192, 192, device2)             # (D',H',W',3)
        grid = grid.view(-1, 3)                                   # (N,3)
        # grid = grid.unsqueeze(0).repeat(B, 1, 1)                  # (B,N,3)
        grid = grid.unsqueeze(0)

        with ClearCache():
            with torch.no_grad():
                try:
                    output_preds = model.forward(grid)
                except RuntimeError:
                    grid = grid.to('cpu')
                    output_preds = model.to('cpu').forward(grid)
                print("Inference : device =", grid.device)
        
        output_preds = list(output_preds)
        for i in range(len(output_preds)):
            output_preds[i] = output_preds[i].to('cpu') #moving all outputs to 'cpu'
        
        return output_preds






'''

# old
class ModelTrainer(nn.Module):
    def __init__(self, config, lf_gt, hf_ground_truth, prior_seg_dice, lf_gt_seg_dice, M):
        super(ModelTrainer, self).__init__()
        self.device = get_device()
        self.M = M
        config["in_features"] = 256 if(config["ffe"]==True) else  3
        print("Features = ", config["in_features"])
        self.config = config
        self.model = get_model(config).to(self.device)
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
        # self.model_input, self.ground_truth = next(iter(lf_dataloader))
        # self.model_input = self.model_input.to(self.device)
        
        self.ground_truth = torch.from_numpy(lf_gt)
        self.ground_truth = self.ground_truth.to(self.device)
        # self.model_input = generate_3d_grid(high_res_shape).to(device)

        self.optim = torch.optim.Adam(lr=self.lr, params=self.model.parameters())#, weight_decay=0.05) #0.05 default
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = self.scheduler_step_size, gamma=self.scheduler_gamma)
        self.hf_ground_truth =  torch.from_numpy(hf_ground_truth).to(self.device)
        self.prior_seg_dice = prior_seg_dice.to(self.device)
        self.lf_gt_seg_dice = lf_gt_seg_dice.to(self.device)
        self.l1, self.l2, self.l3, self.l4, self.l5 = config["l1"], config["l2"], config["l3"], config["l4"], config["l5"]
        print("init complete")

    def compute_loss(self, lf_output, model_output_seg, model_output_seg_pre, model_output_img, lf_gt, prior_seg_dice, lf_gt_seg_dice ):
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5
        print("output shapes : ", lf_output.shape,lf_gt.shape, model_output_img.shape) 
        model_output_seg_dice = torch.stack((model_output_seg[:,:,3].reshape(self.size), model_output_seg[:,:,0].reshape(self.size), model_output_seg[:,:,1].reshape(self.size), model_output_seg[:,:,2].reshape(self.size)), dim=0).unsqueeze(0)
        lf_pred_seg_dice = torch.stack(((model_output_seg[:,:,3].reshape(self.size))[::2, ::2], (model_output_seg[:,:,0].reshape(self.size))[::2, ::2], (model_output_seg[:,:,1].reshape(self.size))[::2, ::2], (model_output_seg[:,:,2].reshape(self.size))[::2, ::2]), dim=0).unsqueeze(0)
        
        
        L_mse = l1 * ((lf_output - lf_gt)**2).mean() #MSE Loss
        L_prior = l2 * (self.dice(model_output_seg_dice, prior_seg_dice))
        L_seg= l3 * (self.dice(lf_pred_seg_dice, lf_gt_seg_dice))
        TV_seg = ((l4[0] * total_variation(model_output_seg_pre[0,:,0].view(self.size), reduction='mean')) + (l4[1]* total_variation(model_output_seg_pre[0,:,1].view(self.size), reduction='mean')) + (l4[2] * total_variation(model_output_seg_pre[0,:,2].view(self.size), reduction='mean')) + (l4[3] * total_variation(model_output_seg_pre[0,:,3].view(self.size), reduction='mean')))/4.0
        TV_img = ((l5[0] * total_variation(model_output_img[0,:,0].view(self.size), reduction='mean')) +  (l5[1] * total_variation(model_output_img[0,:,1].view(self.size), reduction='mean')) + (l5[2] * total_variation(model_output_img[0,:,2].view(self.size), reduction='mean')) + (l5[3] * total_variation(model_output_img[0,:,3].view(self.size), reduction='mean')))
        loss = L_mse + L_prior + L_seg + TV_seg + TV_img #+ grad_reg_seg + grad_reg_img #Total Loss
        return loss

    def train_batch(self):
        batch_size, num_epochs = 4096, 10
        optimizer = self.optim
        scheduler = self.scheduler
        all_coords = self.model_input
        print("All Coords shape : ", all_coords.shape)
        num_coords = all_coords.shape[1]
        for epoch in tqdm(range(num_epochs)):
            permutation = torch.randperm(num_coords)
            total_loss = 0
            for i in range(0, num_coords, batch_size):
                print("Iteration : ",i, num_coords, batch_size)
                indices = permutation[i:i + batch_size]
                batch_coords = self.model_input[indices]
                batch_lf_gt = self.ground_truth[indices]
                batch_prior_seg_dice = self.prior_seg_dice[indices]
                batch_lf_gt_seg_dice = self.lf_gt_seg_dice[indices]
                print("Test")
                optimizer.zero_grad()
                # predicted_pixels = inr_model(batch_coords)
                model_output_seg_pre, model_output_seg, model_output_img, coords = self.model(batch_coords)
                lf_output = self.phi.forward(model_output_seg, model_output_img, self.M)
                # loss = nn.functional.mse_loss(predicted_pixels, batch_pixels)
                
                loss = self.compute_loss(lf_output, model_output_seg, model_output_seg_pre, model_output_img, batch_lf_gt, batch_prior_seg_dice, batch_lf_gt_seg_dice)
                loss.backward()
                # torch.cuda.synchronize()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item() * batch_coords.shape[0]
            avg_loss = total_loss / num_coords
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.6f}")
            losses, train_metrics = None, None
        return model_output_seg, model_output_img, lf_output, losses, train_metrics


    def train_full(self): 
        #so far used for 2D only
        # hf_ground_truth = self.hf_ground_truth
        prior_seg_dice = self.prior_seg_dice
        lf_gt_seg_dice = self.lf_gt_seg_dice

        losses = {"mse" : [], "prior" : [], "seg" : [], "TV_seg" : [], "TV_img" : [] , "Total_Loss" : []} #Dictionary to store losses
        train_metrics = {"psnr" : [], "ssim": []}
        
        hf_gt_normalized = (norm(self.hf_ground_truth)).unsqueeze(0).unsqueeze(0)
        

        for step in tqdm(range(self.epochs)):
            l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5
            
            model_output_seg_pre, model_output_seg, model_output_img, coords = self.model(self.model_input)
            print("Outputs shape = ", model_output_seg_pre.shape, model_output_seg.shape, model_output_img.shape)
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
            TV_img = ((l5[0] * total_variation(model_output_img[0,:,0].view(self.size), reduction='mean')) +  (l5[1] * total_variation(model_output_img[0,:,1].view(self.size), reduction='mean')) + (l5[2] * total_variation(model_output_img[0,:,2].view(self.size), reduction='mean')) + (l5[3] * total_variation(model_output_img[0,:,3].view(self.size), reduction='mean')))
            loss = L_mse + L_prior + L_seg  + TV_seg + TV_img #+ grad_reg_seg + grad_reg_img #Total Loss
            
            losses["mse"].append(L_mse.item())
            losses["prior"].append(L_prior.item())
            losses["seg"].append(L_seg.item())
            losses["TV_seg"].append(TV_seg.item())
            losses["TV_img"].append(TV_img.item())
            losses["Total_Loss"].append(loss.item())
            
            
            #Metrics
            hf_int_res_normalized = norm(get_full_img(model_output_seg, model_output_img, self.config)).unsqueeze(0).unsqueeze(0).to(self.device)
            train_metrics['psnr'].append(self.psnr(hf_gt_normalized, hf_int_res_normalized).item())
            train_metrics['ssim'].append(self.ssim(hf_gt_normalized, hf_int_res_normalized).item())
            torch.cuda.synchronize()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.scheduler.step()

        return model_output_seg, model_output_img, lf_output, losses, train_metrics

    def train_step(self):
   
        optimizer = self.optim
        model = self.model

        high_res_shape = (174, 192, 192)
        low_res_shape  = (87, 96, 96)
        target_lr = self.ground_truth

    
        # coords_hr = self.model_input
        coords_hr = generate_3d_grid(high_res_shape).to(self.device)

        N = coords_hr.shape[0]
        output_list = []
        batch_size = 65536*4  # Adjust based on GPU memory
        model.train()
        
        
        # Batch process high-res coords
        for i in tqdm(range(0, N, batch_size)):
            coord_batch = coords_hr[i:i+batch_size]
            pred_batch = model(coord_batch)
            output_list.append(pred_batch)

            
        print("batch-processing complete")
        # [N_hr, 1] → [174,192,192]
        output_hr = torch.cat(output_list, dim=0).view(*high_res_shape)

        # Downsample from (174,192,192) → (87,96,96)
        # Add batch & channel dims: [1,1,174,192,192]
        output_ds = F.interpolate(output_hr.unsqueeze(0).unsqueeze(0),
                                size=low_res_shape,
                                mode='trilinear',
                                align_corners=False).squeeze()

        # Compute loss
        loss = F.mse_loss(output_ds.view(*low_res_shape), target_lr.view(*low_res_shape))
        print(loss)
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

'''

"""

from Models.models import Siren, Finer
from Utils.utils import get_full_img, norm, get_device, dice_stack_helper, get_model
from Utils.defaults import default_config
from LFSynth.ContrastModulation import ContrastModulation
from Models.model_trainer_3D import ModelTrainer
from Data.load_data_3d import load_data, get_gt_seg
config = default_config
config["ffe"] = False
config["in_features"] = 3 #3D input
model = get_model(config).to(get_device())
# config["model"] = 0 #0 for SIREN else FINER
hf_ground_truth, lf_dataloader, prior_seg_dice, lf_gt_seg_dice, M = load_data(1, config)
print("Shapes : HF = ", hf_ground_truth.shape, "prior = ",prior_seg_dice.shape, "lf seg = ", lf_gt_seg_dice.shape)

test_trainer = ModelTrainer(config, lf_dataloader, hf_ground_truth, prior_seg_dice, lf_gt_seg_dice, M)
for step in range(50):
    loss = test_trainer.train_step()
    if step % 10 == 0:
        print(f"[Step {step}] Loss = {loss:.6f}")

"""
