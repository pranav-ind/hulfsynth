from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi
import torch.nn as nn
from kornia.losses import total_variation
from monai.losses import DiceLoss, DiceCELoss, SoftclDiceLoss, DiceFocalLoss, NACLLoss
from tqdm import tqdm, tqdm_notebook
from sklearn.metrics import mean_squared_error
from monai.losses.dice import *  # NOQA
from IPython.display import clear_output
from monai import metrics
import monai
from matplotlib import pyplot as plt
import tempfile, os
import copy
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm, tqdm_notebook
import wandb



from Models.models import Siren, Finer
from Utils.utils import get_full_img, norm, get_device, dice_stack_helper, get_model, ClearCache
from Data.load_data_3d import load_data, get_gt_seg
from Utils.defaults import default_config
from Utils.plotting_utils2 import plot_seg_results_paper, plot_final_results_paper, plot_hf_results_paper
from Utils.plotting_utils import loss_plot, plot_image_metrics, plot_4_images
from LFSynth.ContrastModulation import ContrastModulation


def define_coords(img_shape) -> torch.Tensor:
    """
    defines coordinate between -1 to 1 of shape ndims
    returns tensor of shape (*img_shape, ndims)
    """
    ndims = len(img_shape)
    coords = [torch.linspace(-1, 1, img_shape[i])
              for i in range(ndims)]

    coords = torch.meshgrid(*coords, indexing=None)
    coords = torch.stack(coords, dim=ndims)

    return coords


class CoordsPatch(Dataset):
    #Adapated from INRMorph - https://github.com/aisha-lawal/inrmorph/blob/506b43125150a8671f54d9e23d3f27c3a85e43dc/data_modules/inrmorph.py
    def __init__(self, patch_size, num_patches, image):
        # super(self, CoordsPatch).__init__()
        self.patch_size = np.ceil(np.array(patch_size) / 2).astype(np.int16)
        
        self.ndims = len(self.patch_size)
        self.image = image
        self.img_shape = self.image.shape
        self.coords = define_coords(self.img_shape)
        # print(self.coords.shape)
        self.dx = torch.div(2, torch.tensor(self.coords.shape[:-1]))
        # print(self.dx)
        self.num_patches = num_patches  # how many random patches to sample
        # self.set_seed = set_seed(42)

        self.patch_size = np.ceil(np.array(patch_size) / 2).astype(np.int16)
        # print("patch size", self.patch_size)
        patch_dx_dims = torch.tensor(self.patch_size) * self.dx
        

        patch_coords = [torch.linspace(-patch_dx_dims[i], patch_dx_dims[i], 2 * self.patch_size[i]) for i in range(self.ndims)]
        # print("path coords : ", len(patch_coords), patch_coords[0].shape, patch_coords[1].shape, patch_coords[2].shape )

        patch_coords = torch.meshgrid(*patch_coords, indexing=None)
        self.patch_coords = torch.stack(patch_coords, dim=self.ndims)

        coords = self.coords[self.patch_size[0]:-self.patch_size[0],
                 self.patch_size[1]:-self.patch_size[1], self.patch_size[2]:-self.patch_size[2], ...] #need to re-check this
        # print("coords : ", coords.shape)
        self.spatial_size = coords.shape[:-1]
        # print("spatial size : ", self.spatial_size)
        self.coords = coords
        

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        
        indx = np.random.randint(0, np.prod(self.spatial_size))
        inds = np.unravel_index(indx, self.spatial_size)


        center = self.coords[inds[0], inds[1], inds[2], :]
        coords = torch.clone(self.patch_coords)
        


        coords[..., 0] = coords[..., 0] + center[0]
        coords[..., 1] = coords[..., 1] + center[1]
        coords[..., 2] = coords[..., 2] + center[2]
        return coords


#helper functions : chunk_size = (86, 96, 96) and lf = (43, 48, 96)


def lf_seg_loss(model_output_seg, chunk_size):
    
    lf_pred_seg_dice = torch.stack((
    (model_output_seg[:,:,3].reshape(chunk_size))[::2, ::2], 
    (model_output_seg[:,:,0].reshape(chunk_size))[::2, ::2], 
    (model_output_seg[:,:,1].reshape(chunk_size))[::2, ::2], 
    (model_output_seg[:,:,2].reshape(chunk_size))[::2, ::2]), dim=0).unsqueeze(0)

    return lf_pred_seg_dice

def model_seg_loss(model_output_seg, chunk_size):
    return torch.stack((
    model_output_seg[:,:,3].reshape(chunk_size),
    model_output_seg[:,:,0].reshape(chunk_size),
    model_output_seg[:,:,1].reshape(chunk_size),
    model_output_seg[:,:,2].reshape(chunk_size)), dim=0).unsqueeze(0)
    
class ModelTrainer(nn.Module):
    def __init__(self, config, lf_gt, prior_seg_dice, lf_gt_seg_dice, M):
        super(ModelTrainer, self).__init__()
        self.device = get_device()
        # self.device = 'cpu'
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
        
        
        self.chunk = (192*172*192)//64 #init chunk via config
        # self.patch_size = [48, 44, 48] #init patch_size via config
        # self.patch_size = [96, 86, 96] #init patch_size via config
        self.patch_size = [48, 44, 48] #init patch_size via config
        self.patch_size_lf = (self.patch_size[0], self.patch_size[1]//2, self.patch_size[2]//2) #init chunk_size via config
        self.num_patches_train = 80 #init num_patches_train via config
        self.hf_size = (192, 172, 192) #init hf_size via config
        self.lf_size = (192, 88, 96) #init lf_size via config


        self.dice = DiceCELoss(include_background=True,squared_pred = True, reduction='mean', jaccard=False)
        self.psnr = monai.metrics.PSNRMetric(max_val = 1.0)
        self.ssim = monai.metrics.regression.SSIMMetric(spatial_dims=2, data_range=1.0)
        self.phi = ContrastModulation()
        # self.model_input, self.ground_truth = next(iter(lf_dataloader))
        # self.model_input = self.model_input.to(self.device)
        
        self.ground_truth = torch.from_numpy(lf_gt).to(torch.float32)
        self.ground_truth = self.ground_truth.to(self.device)
        # self.model_input = generate_3d_grid(high_res_shape).to(device)

        
        # self.optim = torch.optim.Adam(lr=self.lr, params=self.model.parameters())#, weight_decay=0.05) #0.05 default
        self.optim = torch.optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=0.5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = self.scheduler_step_size, gamma=self.scheduler_gamma)
        # self.hf_ground_truth =  torch.from_numpy(hf_ground_truth).to(self.device)
        self.prior_seg_dice = prior_seg_dice.to(self.device)
        self.lf_gt_seg_dice = lf_gt_seg_dice.to(self.device)
        self.l1, self.l2, self.l3, self.l4, self.l5 = config["l1"], config["l2"], config["l3"], config["l4"], config["l5"]
        print("init complete mt")

    def compute_loss(self, pred_lf_chunk, targ_chunk, model_output_seg, targ_seg_chunk, targ_prior_chunk, model_output_seg_pre, model_output_img):

        chunk_size = self.patch_size
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5
        # print("output shapes : ", lf_output.shape,lf_gt.shape, model_output_img.shape) 
        
        #MSE
        L_mse = l1 * F.mse_loss(pred_lf_chunk, targ_chunk.float()) #/ n_chunks  # normalize
        
        
        #Segmentation
        lf_pred_seg_dice = lf_seg_loss(model_output_seg, chunk_size)
        # print("lf_pred_seg_dice", lf_pred_seg_dice.shape, "targ_seg_chunk", targ_seg_chunk.shape)
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
        # L_prior, L_seg, TV_seg, TV_img = None, None, None, None
        return loss, L_mse, L_prior, L_seg, TV_seg, TV_img

    def train_step(self, losses, coord_input_loader, model: nn.Module, target_gt: torch.Tensor, target_seg: torch.Tensor, target_prior: torch.Tensor):

        chunk_size = self.patch_size
        chunk_size_lf = self.patch_size_lf
        chunk = self.chunk

        
        loss_per_epoch = 0.0
        mse_per_epoch = 0.0
        seg_per_epoch = 0.0
        prior_per_epoch = 0.0
        tv_seg_per_epoch = 0.0
        tv_img_per_epoch = 0.0
        print("executing train step fn....")
        for idx, patch_grid in (enumerate(coord_input_loader)):
            # coord_chunk = patch_grid #shape(1, 48, 43, 48, 3)
            patch_grid = patch_grid.to(self.device)
            target_patch = F.grid_sample(target_gt.unsqueeze(0).unsqueeze(0), patch_grid, mode='bilinear')[:,:,:,::2,::2] #output : (1,1, 48,22,24) #for 5D-input, bilinear == trilinear 
            target_seg_patch = F.grid_sample(target_seg, patch_grid, mode='bilinear')[:,:,:,::2,::2] #output : (1,4, 48,22,24)
            target_prior_patch = F.grid_sample(target_prior, patch_grid, mode='bilinear') #output : (1,4, 48,43,48)
            # print(target_prior_patch.shape, patch_grid.shape)

            coord_chunk = patch_grid.reshape(-1,3).unsqueeze(0) #(1, *chunk,3)
            targ_prior_chunk = target_prior_patch.reshape(1, 4, chunk_size[0]*chunk_size[1]*chunk_size[2]).unsqueeze(-1)  #output : (1,4, *chunk, 1)
            
            targ_chunk = target_patch.reshape(-1, 1).unsqueeze(0)
            targ_seg_chunk = target_seg_patch.reshape(1, 4, chunk_size_lf[0]*chunk_size_lf[1]*chunk_size_lf[2]).unsqueeze(-1) #output : (1,4, *chunk_lf, 1)

            model_output_seg_pre, model_output_seg, model_output_img_pre , model_output_img, coords = model(coord_chunk)
            # print('model_output_img = ', model_output_img.shape, model_output_seg.shape)
            pred_lf_chunk = self.phi.forward(chunk_size, model_output_seg, model_output_img, self.M) #shape (1,1,*chunk_lf)
            # print('pred_lf_chunk', pred_lf_chunk.shape, targ_chunk.shape)
            # print('model_output_seg', model_output_seg.shape)
            loss, L_mse, L_prior, L_seg, TV_seg, TV_img = self.compute_loss(pred_lf_chunk, targ_chunk, model_output_seg, targ_seg_chunk, targ_prior_chunk, model_output_seg_pre, model_output_img) #total_loss per chunk
            loss.backward() 
            
            #visualize observed patches 
            # wandb.log({
            # "seg_patch": wandb.Image((targ_seg_chunk[:,2].view(chunk_size_lf[0], chunk_size_lf[1], chunk_size_lf[2]))[5].detach().cpu().numpy(), mode='L'), #random slice=5
            # "seg_pred_patch": wandb.Image((model_output_seg[:,2].view(chunk_size_lf[0], chunk_size_lf[1], chunk_size_lf[2]))[5].detach().cpu().numpy(), mode='L') #random slice=5,
            
            # # "hf_int": wandb.Image(hf_int_res_normalized.squeeze(0).squeeze(0).detach().cpu().numpy(), mode='L'), 
            # # "lf_op": wandb.Image(lf_output.view(size_lf).detach().cpu().numpy(), mode='L') 
            # })
            
            loss_per_epoch += loss #loss per epoch
            mse_per_epoch += L_mse 
            seg_per_epoch += L_seg
            prior_per_epoch += L_prior
            tv_seg_per_epoch += TV_seg
            tv_img_per_epoch += TV_img
            

            self.coord_chunk = coord_chunk #to save model
            print(self.coord_chunk.shape, coord_chunk.shape)

        return model, loss_per_epoch, mse_per_epoch, seg_per_epoch, prior_per_epoch, tv_seg_per_epoch, tv_img_per_epoch
    

    def train_inr(self):
        model = self.model.to(self.device).train() #set the model to train mode
        device = self.device
        
        
        hf_size = self.hf_size
        lf_size = self.lf_size
        patch_size = self.patch_size
        num_patches_train = self.num_patches_train
        B, _, D, H, W = 1,1, hf_size[0], hf_size[1], hf_size[2] #set this via config

        IO = torch.randn(hf_size, device = device) #index image
        #input 3D-coord grid
        
        coord_input_loader = DataLoader(dataset=CoordsPatch(patch_size=patch_size, num_patches=num_patches_train, image=IO), batch_size=1, shuffle=False, num_workers=1, drop_last=True)
        # coord_input_loader_lf = DataLoader(dataset=CoordsPatch(patch_size=patch_size_lf, num_patches=num_patches_train, image=IO, is_low_res=False), batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)


        #targets
        lf_gt = self.ground_truth
        lf_gt_seg_dice = self.lf_gt_seg_dice
        prior_seg_dice = self.prior_seg_dice


        #set these shapes via config
        target_gt = F.pad(lf_gt, (0, 0, 0, 0, 0, 1)).to(device).permute(2,0,1) #shape : [192, 88, 96]
        target_seg = F.pad(lf_gt_seg_dice, (0, 0, 0, 0, 0, 1)).to(device).permute(0,1,4,2,3) #shape : [1, 4, 192, 88, 96]
        target_prior = prior_seg_dice[:,:,1:-1,:,:].to(device).permute(0,1,4,2,3) #shape : [1, 4, 192, 172, 192]
        # IO = torch.arange(192*172*192).reshape(192, 172, 192).to(device)  # Example index image
        
        
        #Model settings
        optimizer = self.optim
        
        
        losses = {"mse" : [], "prior" : [], "seg" : [], "TV_seg" : [], "TV_img" : [] , "Total_Loss" : []} #Dictionary to store losses
        
        for ep in tqdm(range(self.epochs)):
            optimizer.zero_grad(set_to_none=True)
            
            model, total_loss, mse_per_epoch, seg_per_epoch, prior_per_epoch, tv_seg_per_epoch, tv_img_per_epoch = self.train_step(losses, coord_input_loader, model, target_gt, target_seg, target_prior)
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
            
            # uncomment to log losses in wandb
            wandb.log({"total_loss": losses["Total_Loss"][-1],
            "mse": losses["mse"][-1], 
            "prior": losses["prior"][-1], 
            "seg": losses["seg"][-1], 
            "tv_seg": losses["TV_seg"][-1], 
            "tv_img": losses["TV_img"][-1], 
            })
            
        
        return model, losses
    def inference(self, model):
        device2 = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        model = model.to(device2)
        
        #TBD : init. grid via config
        # grid = make_grid_3d(174, 192, 192, device2)             # (D',H',W',3)
        grid = define_coords((192, 174, 192))
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






def visualize_volume_slices(volume1, volume2, axis=0, num_slices=16, title1='Volume 1', title2='Volume 2'):
    """
    Visualizes `num_slices` slices from two 3D volumes side-by-side.
    
    Args:
        volume1 (torch.Tensor): 3D tensor (D, H, W)
        volume2 (torch.Tensor): 3D tensor (D, H, W)
        axis (int): Axis along which to slice (0=z, 1=y, 2=x)
        num_slices (int): Number of slices to show (default=16)
        title1 (str): Title for first volume
        title2 (str): Title for second volume
    """

    # assert volume1.shape == volume2.shape, "Volumes must be the same shape"
    assert volume1.dim() == 3, "Volumes must be 3D tensors"

    # Select slice indices (evenly spaced)
    dim = volume1.shape[axis]
    step = max(dim // num_slices, 1)
    slice_indices = list(range(0, dim, step))[:num_slices]

    fig, axes = plt.subplots(4, num_slices // 2, figsize=(16, 8))
    axes = axes.flatten()

    for i, idx in enumerate(slice_indices):
        if axis == 0:
            slice1 = volume1[idx, :, :].cpu().numpy()
            slice2 = volume2[idx, :, :].cpu().numpy()
        elif axis == 1:
            slice1 = volume1[:, idx, :].cpu().numpy()
            slice2 = volume2[:, idx, :].cpu().numpy()
        elif axis == 2:
            slice1 = volume1[:, :, idx].cpu().numpy()
            slice2 = volume2[:, :, idx].cpu().numpy()
        else:
            raise ValueError("Invalid axis")

        axes[i].imshow(slice1, cmap='gray')
        axes[i].set_title(f'{title1} - Slice {idx}')
        axes[i].axis('off')

        # Add the second volume slice in the next row (offset by num_slices)
        axes[i + num_slices].imshow(slice2, cmap='gray')
        axes[i + num_slices].set_title(f'{title2} - Slice {idx}')
        axes[i + num_slices].axis('off')

    plt.tight_layout()
    # plt.show()
    return fig
    



if __name__ == '__main__':
    wandb.login()
    config = copy.deepcopy(default_config)
    config["ffe"] = False
    config["in_features"] = 3 #3D input
    config["lr"] = 1e-3
    config["l1"] = 2.5
    config["l2"] = 1e-8
    config["l3"] = 1.0
    config["l4"] =  [5e-3, 5e-3, 5e-3, 5e-3]
    config["l5"] = [5e-2, 5e-2, 5e-3, 9e-2]
    config["w0"] = 30

    config["total_steps"] = 1

    # model = get_model(config).to(get_device())
    hf_ground_truth, lf_gt, prior_seg_dice, lf_gt_seg_dice, M = load_data(1, config) #uncomment
    # print(hf_ground_truth.shape, lf_gt.shape, prior_seg_dice.shape, lf_gt_seg_dice.shape )


    project_ = "hulfsynth_ulfenc"
    # run_name = "run_" + str(run_id)
    run = wandb.init(project=project_)


    trainer = ModelTrainer(config, lf_gt, prior_seg_dice, lf_gt_seg_dice, M) #init
    model, losses = (trainer.train_inr())
    
    model_saving_path =  "./wandb/saved_models/model.onnx"
    dummy_input = torch.randn(1, 101376, 3)
    torch.onnx.export(model, dummy_input, model_saving_path)
    print("locally saved model to: ", model_saving_path)
    wandb.save(model_saving_path)

    run.log_model(path=model_saving_path, name="model")
    run.finish()

    '''
    output_preds = trainer.inference(model)
    for idx, op in enumerate(output_preds):
        torch.save(op, "./temporary/" + str(idx)+ ".pt")
        print("saving to ", "./temporary/" + str(idx)+ ".pt" + "...")



    final_output_seg_pre, final_output_seg, final_output_img, output_coords = output_preds
    final_output_seg_pre = final_output_seg_pre.reshape(192, 174, 192,4)
    final_output_seg = final_output_seg.reshape(192, 174, 192,4)
    final_output_img = final_output_img.reshape(192, 174, 192,4)
    final_output = final_output_seg * final_output_img


    fig1 = visualize_volume_slices(final_output[:,:,:,2], final_output_seg[:,:,:,2], axis=0, num_slices=16, title1='final_img2', title2='final_seg2')
    fig2 = visualize_volume_slices(final_output[:,:,:,1], final_output_seg[:,:,:,1], axis=0, num_slices=16, title1='final_img1', title2='final_seg1')
    fig_loss = loss_plot(losses)

    fig1.savefig("./temporary/fig1.png")
    fig2.savefig("./temporary/fig2.png")
    fig_loss.savefig("./temporary/fig_loss.png")
    torch.cuda.empty_cache()
    '''