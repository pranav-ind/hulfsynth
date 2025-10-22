import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np

import random
from typing import Tuple, List, Optional, Any
import matplotlib.pyplot as plt
from datetime import datetime
import copy
import argparse
import ast
import os
import shutil


import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
import wandb
import pprint

import monai
from monai.losses import DiceLoss, DiceCELoss, SoftclDiceLoss, DiceFocalLoss, NACLLoss
from kornia.losses import total_variation

from Models.model import MLP, initialize_siren_weights, SineLayer, ReLULayer, WIRELayer, initialize_wire_weights
from Models.model_trainer import ModelTrainerModule



from Models.models import Siren, Finer

from Utils.utils import get_full_img, norm, get_device, dice_stack_helper, get_model, ClearCache
from Data.load_ixi import load_data, get_hf_observed_segmentations, load_sensitivity_data
from Utils.defaults import default_config
from Utils.plotting_utils2 import plot_seg_results_paper, plot_final_results_paper, plot_hf_results_paper
from Utils.plotting_utils import loss_plot, plot_image_metrics, plot_4_images
from LFSynth.ContrastModulation import ContrastModulation

from Data.patchwise3D import RandomPointsDataset



def wandb_setup(siren_module):
    wandb.login()
    project_ = "hulfsynth"
    run = wandb.init(project=project_)
    wandb_logger = WandbLogger(project=project_)
    wandb_logger.watch(siren_module, log="all")
    return run, wandb_logger


def parse_list(list_string: str) -> List[Any]:
    try:
        parsed_list = ast.literal_eval(list_string)
        if isinstance(parsed_list, list):
            return parsed_list
        else:
            raise ValueError("Input must represent a list, e.g., '[1, 2, 3]'")
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(
            f"Invalid list format: '{list_string}'. Error: {e}"
        )

def delete_folder_and_contents(folder_path):
    """
    Deletes a folder and all its contents recursively.
    """
    if os.path.exists(folder_path):
        try:
            # shutil.rmtree() stands for "remove tree" and deletes the directory
            # and everything inside it.
            shutil.rmtree(folder_path)
            print(f"Successfully deleted the folder: {folder_path}")
        except OSError as e:
            # Handle permissions errors or other system issues
            print(f"Error deleting folder {folder_path}: {e}")
    else:
        print(f"Folder not found: {folder_path}")




if __name__ == '__main__':
# def main():
    config = copy.deepcopy(default_config)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--id") #sens_id
    parser.add_argument("-dn", "--dn") #dataset_num
    parser.add_argument("-an", "--an") #activation_function
    parser.add_argument("-l1", "--l1")
    parser.add_argument("-l3", "--l3")
    parser.add_argument("-l4", "--l4", )#, type=float, default=[])
    parser.add_argument("-l5", "--l5", )#, type=float, default=[])
    # parser.add_argument("-l4", "--l4", nargs="*", type=float, default=[])
    # parser.add_argument("-l5", "--l5", nargs="*", type=float, default=[])
    parser.add_argument("-ep", "--epochs") #epochs
    args = parser.parse_args()
    
    pl.seed_everything(seed=9600, workers=True)
    config = copy.deepcopy(default_config)

    print("printing args: ", args)
    
    config["in_features"] = 3
    config["l1"] = float(ast.literal_eval(args.l1)) #100 #mse
    config["l3"] = float(ast.literal_eval(args.l3)) #20 #seg
    config["l4"] = ast.literal_eval(args.l4) #0.01 #tv_img
    config["l5"] = ast.literal_eval(args.l5) # 0.1 #tv_seg
    config["epochs"] = int(ast.literal_eval(args.epochs))
    config["dataset_num"] = int(ast.literal_eval(args.dn) ) #102
    config["activation_fn"] = args.an
    config["sens_id"] = int(ast.literal_eval(args.id))


    config["slice"] = 175
    config["is_new_contrast"] = False #make this true when using new c vector
    config["points_num"] = 96*96*4
    config["downsampled_points"] = 48*48*4
    config["hf_chunk_size"] = (96, 96, 4)
    config["lf_chunk_size"] =  (48, 48, 4)
    

    
    dataset_num = config["dataset_num"]
    slice_num = config["slice"]
    sens_id = config["sens_id"]
    POINTS_PER_SAMPLE = config["points_num"]
    lf_points_per_sample = config["downsampled_points"]


    # hf_ground_truth, lf_gt, lf_gt_seg_dice, M = load_data(dataset_num, config) #uncomment
    hf_ground_truth, lf_gt, lf_gt_seg_dice, M = load_sensitivity_data(dataset_num, config, sens_id)
    gt_image = torch.tensor(norm(hf_ground_truth)).unsqueeze(-1)
    gt_image = gt_image.to(torch.float32)
    lf_gt = torch.tensor(norm(lf_gt)).unsqueeze(-1)
    lf_gt = lf_gt.to(torch.float32)
    # print("gt_image: ", gt_image.shape, "lf_gt: ", lf_gt.shape, "lf_gt_seg_dice: ", lf_gt_seg_dice.shape)
    print('gt_image, lf_gt loaded')
    (wm_obs_seg, gm_obs_seg, csf_obs_seg, bg_obs_seg) = get_hf_observed_segmentations(dataset_num, config)
    hf_observed_seg_dice = torch.stack((bg_obs_seg, wm_obs_seg, gm_obs_seg, csf_obs_seg), dim=0).unsqueeze(0)
    
    config["M"] = M
    config["size"] = hf_ground_truth.shape
    config["size_lf"] = lf_gt.shape[:-1]

    dataset = RandomPointsDataset(gt_image, lf_gt, lf_gt_seg_dice, points_num=POINTS_PER_SAMPLE)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False) # We set a batch_size of 1 since our dataloader is already returning a batch of points.
    
    HIDDEN_SIZE = 128 #best_config; 128/5/10000
    NUM_LAYERS = 5
    TRAINING_EPOCHS = config["epochs"]
    LEARNING_RATE = 5e-5
    SIREN_FACTOR = 30.0
    WIRE_OMEGA = 30.0
    WIRE_SIGMA = 10.0
    activation_fn = config["activation_fn"] #either WIRE or SIREN

    
    if(activation_fn == 'SIREN'):
        inr_mlp = MLP(in_size=3, out_size=5, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, layer_class=SineLayer, siren_factor=SIREN_FACTOR,)
        initialize_siren_weights(inr_mlp, SIREN_FACTOR) # Re-initialize the weights and make sure they are different                                
    else:    
        inr_mlp = MLP(in_size=3, out_size=5, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, layer_class=WIRELayer,  wire_omega=WIRE_OMEGA, wire_sigma=WIRE_SIGMA)
        initialize_wire_weights(inr_mlp, WIRE_OMEGA)

    wandb_logger = WandbLogger(project="hulfsynth", config=args)#, id="test_run_1001", resume="allow")
    trainer_module = ModelTrainerModule(wandb_logger = wandb_logger,
                                    network=inr_mlp,
                                    hf_gt_im=gt_image,
                                    lf_gt_im = lf_gt,
                                    lf_gt_seg = lf_gt_seg_dice,
                                    hf_gt_seg = hf_observed_seg_dice,
                                    config = config,
                                    lr=LEARNING_RATE,
                                    name=activation_fn,)
        
    wandb_logger.watch(trainer_module, log="all")
    s = datetime.now()
    trainer = pl.Trainer( max_epochs=TRAINING_EPOCHS, logger=wandb_logger, accelerator='gpu', devices=1, strategy='ddp_find_unused_parameters_true', deterministic=True)
    trainer.fit(trainer_module, train_dataloaders=dataloader)
    print(f"Fitting time: {datetime.now()-s}s.")



    dummy_input, _ , _ = next(iter(dataloader))
    dummy_input = dummy_input.view(-1, dummy_input.shape[-1])
    model_saving_path =  "./wandb/model.onnx"
    torch.onnx.export(trainer_module.network.to('cpu'), dummy_input.to('cpu'), model_saving_path)
    print("locally saved model to: ", model_saving_path)
    # wandb.save(model_saving_path)
    wandb_logger.experiment.log_model(path=model_saving_path, name="model")
    print(config)


    #Cleaning up wandb_logger locally to avoid Disk Quota errors
    # os.system("wandb sync --clean --clean-old-hours 0")
    full_local_dir_path = wandb_logger.experiment.dir
    run_id = wandb_logger.experiment.id
    delete_folder_and_contents(full_local_dir_path[:-5]) #deleting ./wandb/run-timestamp-run_id ; wandb logs
    delete_folder_and_contents('./hulfsynth/' + run_id) #deleting ./hulfsynth/run_id ; checkpoints if they were created
