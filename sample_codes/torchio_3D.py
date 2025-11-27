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
import random
import torchio as tio
from torchio.data import UniformSampler, Queue


from projects.hulfsynth.hulfsynth.Models.old.models import Siren, Finer
from Utils.utils import get_full_img, norm, get_device, dice_stack_helper, get_model, ClearCache
from Data.load_data_3d import load_data, get_gt_seg
from Utils.defaults import default_config
from Utils.plotting_utils2 import plot_seg_results_paper, plot_final_results_paper, plot_hf_results_paper
from Utils.plotting_utils import loss_plot, plot_image_metrics, plot_4_images
from LFSynth.ContrastModulation import ContrastModulation
from test3D import visualize_volume_slices



def make_patch_coords(pw, ph, pd, device):
    # local coords in [-1, 1] over patch grid
    z = torch.linspace(-1.0, 1.0, pd, device=device)
    y = torch.linspace(-1.0, 1.0, ph, device=device)
    x = torch.linspace(-1.0, 1.0, pw, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')  # shape (pd, ph, pw)
    coords = torch.stack([xx, yy, zz], dim=-1)           # (pd, ph, pw, 3)
    coords = coords.view(-1, 3)                          # (N, 3)
    return coords  # CPU or device tensor




if __name__ == '__main__':
    wandb.login()
    config = copy.deepcopy(default_config)
    config["in_features"] = 3
    hf_ground_truth, lf_gt, prior_seg_dice, lf_gt_seg_dice, M = load_data(1, config) #uncomment
    hf_gt_subj = tio.Subject(hf_gt_scalar = tio.ScalarImage(tensor=torch.tensor(torch.from_numpy(norm(hf_ground_truth)).unsqueeze(0), dtype=torch.float32))) #format (C,W,H,D)
    patch_size = (44, 48, 48) #W, H, D
    # patch_size = (172, 192, 192) #W, H, D
    sampler = UniformSampler(patch_size)
    # sampler = tio.WeightedSampler(patch_size, probability_map='brain')
    subjects_dataset = tio.SubjectsDataset([hf_gt_subj])
    queue = Queue(
        subjects_dataset,
        max_length=128,        # max number of items the queue stores
        samples_per_volume=40,
        sampler=sampler,
        num_workers=2)
    loader = DataLoader(queue, batch_size=1)
    device = get_device()
    model = get_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    loss_fn = nn.MSELoss()
    pw, ph, pd = patch_size
    coords_patch = make_patch_coords(pw, ph, pd, device=device)  # (N, 3)
    N = coords_patch.shape[0]
    B = 1 #batch_size
    epochs = 1250
    project_ = "hulfsynth_ulfenc"
    # run_name = "run_" + str(run_id)
    run = wandb.init(project=project_)

    chunk_size = patch_size
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        n_batches = 0
        for idx, batch in enumerate(loader):
            # batch is a dict; images accessible as batch['low'][tio.DATA] and batch['high'][tio.DATA]

            # low_patches = batch['low'][tio.DATA].to(device)   # (B,1,pw,ph,pd)
            high_patches = batch["hf_gt_scalar"][tio.DATA].to(device) # (B,1,pw,ph,pd)

            # B = low_patches.shape[0]
            hf_target = high_patches.view(B, 1, -1).permute(0,2,1)  # (B,*N,1)

            # Make coords batch: (B,N,3)
            coords_batched = coords_patch.unsqueeze(0).expand(B, -1, -1)  # (B,*N,3)

            optimizer.zero_grad()
            # preds = model(coords_batched)  # (B,N,1)
            model_output_seg_pre, model_output_seg, model_output_img_pre , model_output_img, coords = model(coords_batched)
                    
            model_output_img_dice = torch.stack((model_output_img[:,:,3].reshape(chunk_size), model_output_img[:,:,0].reshape(chunk_size), model_output_img[:,:,1].reshape(chunk_size), model_output_img[:,:,2].reshape(chunk_size)), dim=0).unsqueeze(0)
            hf_pred = model_output_img_dice[0,0] + model_output_img_dice[0,1] + model_output_img_dice[0,2] #+ model_output_img_dice[0,3]
            
            loss = loss_fn(hf_pred.flatten(), hf_target.flatten())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            wandb.log({"total_loss": running_loss,
            })

        epoch_loss = running_loss / max(1, n_batches)
    
    model_saving_path =  "./wandb/saved_models/model_supervised.onnx"
    dummy_input = torch.randn(1, 101376, 3)
    torch.onnx.export(model, coords_batched, model_saving_path)
    print("locally saved model to: ", model_saving_path)
    wandb.save(model_saving_path)

    run.log_model(path=model_saving_path, name="model")
    run.finish()

