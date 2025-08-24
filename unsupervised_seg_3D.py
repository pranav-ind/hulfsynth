import torch
from torch import nn

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

import random
import numpy as np

from typing import Tuple, List, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime



from torch import nn
import lightning as pl
import monai
from monai.losses import DiceLoss, DiceCELoss, SoftclDiceLoss, DiceFocalLoss, NACLLoss



from Models.models import Siren, Finer
from Utils.utils import get_full_img, norm, get_device, dice_stack_helper, get_model, ClearCache
from Data.load_data_3d import load_data, get_gt_seg
from Utils.defaults import default_config
from Utils.plotting_utils2 import plot_seg_results_paper, plot_final_results_paper, plot_hf_results_paper
from Utils.plotting_utils import loss_plot, plot_image_metrics, plot_4_images
from LFSynth.ContrastModulation import ContrastModulation
from test3D import visualize_volume_slices
import copy

import wandb
from lightning.pytorch.loggers import WandbLogger

POINTS_PER_SAMPLE = 96*96*4
lf_points_per_sample = 48*48*4
class ReLULayer(nn.Module):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.relu(x)
        return x

class MLP(nn.Module):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 layer_class: nn.Module = ReLULayer,
                 **kwargs):
        super().__init__()

        a = [layer_class(in_size, hidden_size, **kwargs)]
        for i in range(num_layers - 1):
            a.append(layer_class(hidden_size, hidden_size, **kwargs))
        
        # a.append(nn.Linear(hidden_size, 1)) #actual image
        a.append(nn.Linear(hidden_size, out_size)) #segmentations+image
        self.layers = nn.ModuleList(a)        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        output_image_pre = x[:,0] #output image before applying activation function
        output_seg_pre = x[:,1:] #output seg before applying activation function
        output_image = self.relu(output_image_pre)
        output_seg = self.softmax(output_seg_pre)
        return output_image, output_seg





class RandomPointsDataset(Dataset):
    def __init__(self, image: torch.Tensor, lf_image:torch.Tensor, lf_gt_seg_dice:torch.Tensor, points_num: int = POINTS_PER_SAMPLE):
        super().__init__()
        self.device = get_device()
        self.points_num = points_num
        assert image.dtype == torch.float32
        assert lf_image.dtype == torch.float32
        self.image = image.to(self.device)  # (H, W, ..., C)
        self.lf_image = lf_image.to(self.device)  # (H, W, ..., C)
        self.lf_gt_seg_dice = lf_gt_seg_dice.permute(1,2,3,4,0).to(self.device) #(tissues, H,W,D, C)
        # self.dim_sizes = self.image.shape[:-1]  # Size of each spatial dimension
        self.dim_sizes = self.lf_image.shape[:-1]  # Size of each spatial dimension
        

        # To help us define the input/output sizes of our network later
        # we store the size of our input coordinates and output values
        self.coord_size = len(self.image.shape[:-1])  # Number of spatial dimensions
        self.value_size = self.lf_image.shape[-1]  # Channel size
        # self.value_size = self.lf_image.shape[-1]  # Channel size

    def __len__(self):
        return 1


    def __getitem__(self, idx: int):
        # Create random sample of pixel indices
        
        point_indices = [torch.randint(0, i, (self.points_num,), device=self.device) for i in self.dim_sizes]
        # print(point_indices[0].shape)
        # Retrieve image values from selected indices
        point_values = self.lf_image[tuple(point_indices)]

        point_values_seg = [self.lf_gt_seg_dice[i][tuple(point_indices)] for i in range(self.lf_gt_seg_dice.shape[0])]
        point_values_seg = torch.stack(point_values_seg,axis = 0)
        # print(point_values.shape, point_values_seg.shape)
        point_values = F.interpolate(point_values.unsqueeze(0).unsqueeze(0).squeeze(-1), scale_factor=0.25).squeeze(0).squeeze(0) #downsampling lf_gt
        point_values_seg = [F.interpolate(point_values_seg[i].unsqueeze(0).unsqueeze(0).squeeze(-1), scale_factor=0.25).squeeze(0).squeeze(0) for i in range(self.lf_gt_seg_dice.shape[0])] #downsampling lf_gt_seg
        point_values_seg = torch.stack(point_values_seg,axis = 0)
        # print(point_values.shape, point_values_seg.shape)

        # Convert point indices into normalized [-1.0, 1.0] coordinates
        point_coords = torch.stack(point_indices, dim=-1)
        spatial_dims = torch.tensor(self.dim_sizes, device=self.device)
        point_coords_norm = point_coords / (spatial_dims / 2) - 1

        # The subject index is also returned in case the user wants to use subject-wise learned latents
        return point_coords_norm, point_values, point_values_seg





# We will track visual results every few epochs and visualize them after training
def plot_reconstructions(progress_ims: List[Tuple[int, torch.Tensor]], gt_im: torch.Tensor):
    ncols = len(progress_ims) + 1
    fig_width = 5
    fig, axs = plt.subplots(ncols=ncols, figsize=(ncols*fig_width, fig_width))
    # Plot all reconstructions images predicted by the model
    for i, (epoch, im, metric) in enumerate(progress_ims):
        im = im.cpu().numpy()
        ax = axs[i]
        ax.imshow(im, cmap='gray')
        ax.axis('off')
        title = f'Epoch: {epoch}, PSNR: {metric}'
        ax.set_title(title)
    # PLot ground-truth image
    gt_im = gt_im.cpu().numpy()
    axs[-1].imshow(gt_im, cmap='gray')
    axs[-1].axis('off')
    axs[-1].set_title('Ground Truth')
    plt.tight_layout()
    plt.show()

# We will also track the PSNR of our training samples
def psnr(pred, ref):
    max_value = ref.max()
    mse = torch.mean((pred - ref) ** 2, dim=(-2, -1))
    out = 20 * torch.log10(max_value / torch.sqrt(mse))
    return out.mean()

# Let's create a function to plot our psnr scores throughout training
def plot_scores(models: List['INRModule']):
    fig, ax = plt.subplots()
    # For each model, plot list of scores
    for model in models:
        epochs, scores = [i for i, _ in model.scores], [v for _, v in model.scores]
        ax.plot(epochs, scores, label=model.name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR')
    ax.set_title('PSNR over epochs')
    ax.legend()
    # plt.show()
    return fig




class INRLightningModule(pl.LightningModule):
    def __init__(self,
                 network: MLP,
                 gt_im: torch.Tensor,
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
        self.gt_im = gt_im
        self.eval_interval = eval_interval
        self.visualization_intervals = visualization_intervals
        self.progress_ims = []
        self.scores = []
        self.dice = DiceCELoss(include_background=True,squared_pred = True, reduction='mean', jaccard=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def forward(self, coords):
        return self.network(coords)

    def training_step(self, batch, batch_idx):
        coords, values, values_seg = batch
        coords = coords.view(-1, coords.shape[-1])
        values = values.view(-1, values.shape[-1])
        # values_seg = values_seg.view(-1, values_seg.shape[-1])
        # values_lf = values_lf.view(-1, values_lf.shape[-1])
        print("true_values: ",coords.shape, values.shape, values_seg.shape)
        
        output_image, output_seg = self.forward(coords)
        print("pred: ", output_image.shape, output_seg.shape)
        outputs_lf = F.interpolate(output_image.unsqueeze(0).unsqueeze(0).squeeze(-1), scale_factor=0.25).squeeze(0).squeeze(0)
        
        pred_seg = [(F.interpolate(output_seg[:,i].unsqueeze(0).unsqueeze(0), scale_factor=0.25).squeeze(0).squeeze(0)).reshape(48,48,4) for i in range(output_seg.shape[-1])] #downsampling lf_gt_seg
        pred_seg = torch.stack(pred_seg,axis = 0).unsqueeze(0) # shape(1,4 48, 48, 4)
        values_seg = [values_seg[:,i].reshape(48,48,4) for i in range(values_seg[0].shape[0])]
        values_seg = torch.stack(values_seg,axis = 0).unsqueeze(0) # shape(1,4 48, 48, 4)
        # print("seg: ", pred_seg.shape, values_seg.shape)
        dice_loss = self.dice(pred_seg, values_seg)
        mse_loss = nn.functional.mse_loss(outputs_lf, values) * 10 

        loss = dice_loss + mse_loss
        print("loss: ", loss.item(), dice_loss.item(), mse_loss.item())
        # loss = loss1 + loss2
        '''
        wandb.log({"total_loss": loss.item(),
            "mse": losses["mse"][-1], 
            "prior": losses["prior"][-1], 
            "seg": losses["seg"][-1], 
            "tv_seg": losses["TV_seg"][-1], 
            "tv_img": losses["TV_img"][-1], 
            })
        '''
        print(loss.item())
        pred_im  = self.sample_at_resolution(self.gt_im.shape[:-1])
        # print("pred_img: ", pred_im.shape)
        wandb_logger.log_image(key="pred", images=[norm(pred_im[:,:,90]).unsqueeze(0), norm(pred_im[:,:,95]).unsqueeze(0)], caption=["slice: 90", "slice: 95"])
        pred_im = pred_im.reshape(self.gt_im.shape)
        psnr_value = psnr(pred_im, self.gt_im.to(pred_im.device)).cpu().item()
        # wandb.log({"total_loss": loss.item(), "psnr": psnr_value, "hf_loss": loss1.item(), "ulf_loss": loss2.item()})
        wandb.log({"total_loss": loss.item(), "psnr": psnr_value, "mse": mse_loss.item(), "seg": dice_loss.item() })
        # self.log("total_loss", loss.item())
        # self.loggt("psnr", psnr_value)
        return loss

    def on_train_epoch_end(self):
        """ At each visualization interval, reconstruct the image using our INR """

        # if (self.current_epoch + 1) % self.eval_interval == 0 or self.current_epoch == 0:
        # pred_im = self.sample_at_resolution(self.gt_im.shape[:-1])
        # pred_im = pred_im.reshape(self.gt_im.shape)
        # psnr_value = psnr(pred_im, self.gt_im.to(pred_im.device)).cpu().item()
        # wandb.log({"psnr": psnr_value})
        # self.scores.append((self.current_epoch + 1, psnr_value))  # Log PSNR
        # if self.current_epoch + 1 in self.visualization_intervals:
        #     self.progress_ims.append((self.current_epoch + 1, pred_im.cpu(), psnr_value))
        pass

    @torch.no_grad()
    def sample_at_resolution(self, resolution: Tuple[int, ...]):
        print("sampling at resolution")
        """ Evaluate our INR on a grid of coordinates in order to obtain an image. """
        meshgrid = torch.meshgrid([torch.arange(0, i, device=self.device) for i in resolution], indexing='ij')
        coords = torch.stack(meshgrid, dim=-1)
        coords_norm = coords / torch.tensor(resolution, device=self.device) * 2 - 1
        coords_norm_ = coords_norm.reshape(-1, coords.shape[-1])
        predictions_, _ = self.forward(coords_norm_)
        predictions = predictions_.reshape(resolution)

        return predictions



class SineLayer(nn.Module):
    """
        Implicit Neural Representations with Periodic Activation Functions
        Implementation based on https://github.com/vsitzmann/siren?tab=readme-ov-file
    """
    def __init__(self, in_size, out_size, siren_factor=30., **kwargs):
        super().__init__()
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        self.siren_factor = siren_factor
        self.linear = nn.Linear(in_size, out_size, bias=True)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sin(self.siren_factor * x)
        return x



import math

def initialize_siren_weights(network: MLP, omega: float):
    """ See SIREN paper supplement Sec. 1.5 for discussion """
    old_weights = network.layers[1].linear.weight.clone()
    with torch.no_grad():
        # First layer initialization
        num_input = network.layers[0].linear.weight.size(-1)
        network.layers[0].linear.weight.uniform_(-1 / num_input, 1 / num_input)
        # Subsequent layer initialization uses based on omega parameter
        for layer in network.layers[1:-1]:
            num_input = layer.linear.weight.size(-1)
            layer.linear.weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
        # Final linear layer also uses initialization based on omega parameter
        num_input = network.layers[-1].weight.size(-1)
        network.layers[-1].weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
        
    # Verify that weights did indeed change
    new_weights = network.layers[1].linear.weight
    assert (old_weights - new_weights).abs().sum() > 0.0

def wandb_setup():
    wandb.login()
    project_ = "hulfsynth_ulfenc"
    # run_name = "run_" + str(run_id)
    run = wandb.init(project=project_)


if __name__ == '__main__':
    wandb_setup()
    wandb_logger = WandbLogger(project="hulfsynth_ulfenc")

    #initialize network
    HIDDEN_SIZE = 256 #working well; 256/5/3000
    NUM_LAYERS = 5
    TRAINING_EPOCHS = 3000
    LEARNING_RATE = 5e-4


    # gt_image.shape
    # temp = torch.rand(172, 192, 192, 1)

    config = copy.deepcopy(default_config)
    config["in_features"] = 3
    hf_ground_truth, lf_gt, prior_seg_dice, lf_gt_seg_dice, M = load_data(1, config) #uncomment
    gt_image = torch.tensor(norm(hf_ground_truth)).unsqueeze(-1)
    gt_image = gt_image.to(torch.float32)
    lf_gt = torch.tensor(norm(lf_gt)).unsqueeze(-1)
    lf_gt = lf_gt.to(torch.float32)
    print('gt_image, lf_gt loaded')


    dataset = RandomPointsDataset(gt_image, lf_gt, lf_gt_seg_dice, points_num=POINTS_PER_SAMPLE)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False) # We set a batch_size of 1 since our dataloader is already returning a batch of points.
    
    # lf_dataset = RandomPointsDataset(lf_gt, points_num=lf_points_per_sample)
    # lf_dataloader = DataLoader(lf_dataset, batch_size=1, num_workers=0, pin_memory=False)
    
    SIREN_FACTOR = 30.0 
    siren_inr = MLP(in_size=3,
                    out_size=5,
                    hidden_size=HIDDEN_SIZE,
                    num_layers=NUM_LAYERS,
                    layer_class=SineLayer, 
                    siren_factor=SIREN_FACTOR,
                    )
    # Re-initialize the weights and make sure they are different
    initialize_siren_weights(siren_inr, SIREN_FACTOR)

    siren_module = INRLightningModule(network=siren_inr,
                                    gt_im=gt_image,
                                    lr=LEARNING_RATE,
                                    name='SIREN',
                                    )




    trainer = pl.Trainer(max_epochs=TRAINING_EPOCHS, logger=wandb_logger)
    
    wandb_logger.watch(siren_module, log="all")



    s = datetime.now()
    trainer.fit(siren_module, train_dataloaders=dataloader)
    print(f"Fitting time: {datetime.now()-s}s.")



    # pred_img_inr = inr_module.sample_at_resolution(gt_image.shape[:-1])
    pred_img_siren = siren_module.sample_at_resolution(gt_image.shape[:-1])


    # plt.imshow(pred_img_inr[:,:,95], cmap='gray')
    # plt.imsave('./results/pred_img_inr.png', pred_img_inr[:,:,95])
    plt.imshow(pred_img_siren[:,:,95], cmap='gray')
    plt.imsave('./results/pred_img_siren.png', pred_img_siren[:,:,95])

    fig = plot_scores([siren_module])
    fig.savefig('./results/psnr.png')
    