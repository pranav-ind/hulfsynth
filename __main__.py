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
import json



import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
import wandb
import pprint

import monai
from monai.losses import DiceLoss, DiceCELoss, SoftclDiceLoss, DiceFocalLoss, NACLLoss
from kornia.losses import total_variation

from Models.model import MLP, initialize_siren_weights, SineLayer, ReLULayer, WIRELayer, initialize_wire_weights, FourierFeatures, initialize_activation_weights
from Models.model_trainer import ModelTrainerModule, PosEncINRLightningModule




from Utils.utils import get_full_img, norm, get_device, dice_stack_helper, ClearCache
from Data.load_ixi import get_hf_observed_segmentations as get_hf_observed_segmentations_ixi, load_sensitivity_data
from Data.load_ixi import load_data as load_ixi_data
from Utils.defaults import default_config
from Utils.plotting_utils2 import plot_seg_results_paper, plot_final_results_paper, plot_hf_results_paper
from Utils.plotting_utils import loss_plot, plot_image_metrics, plot_4_images
from LFSynth.ContrastModulation import ContrastModulation
from LFSynth.ContrastEstimation_val import load_val_data, get_hf_observed_segmentations as get_hf_observed_segmentations_val 
from Data.patchwise3D import RandomPointsDataset



def wandb_setup(siren_module):
    wandb.login()
    project_ = "hulfsynth"
    run = wandb.init(project=project_)
    wandb_logger = WandbLogger(project=project_)
    wandb_logger.watch(siren_module, log="all")
    return run, wandb_logger

def get_data(dataset_num, sens_id):
    ixi_nums = ["102", "105", "127", "128", "130"]
    if dataset_num in ixi_nums:
        if(str(sens_id) !="-1"):
            print("Loading sesitivity data: ", sens_id)
            hf_ground_truth, lf_gt, lf_gt_seg_dice, M = load_sensitivity_data(config)
        else:
            print("Loading IXI data: ", dataset_num)
            hf_ground_truth, lf_gt, lf_gt_seg_dice, M = load_ixi_data(config)
        (wm_obs_seg, gm_obs_seg, csf_obs_seg, bg_obs_seg) = get_hf_observed_segmentations_ixi(config["dataset_num"], config)
        config["slice"] = 175
        config["slice"] = 175 if str(dataset_num) == '102' else (155 if str(dataset_num) == '128' else 160) #slice_num and dataset_num[11: 0011, 24: 0015, 19: others]

    else:
        print("Loading val data: ", dataset_num)
        hf_ground_truth, lf_gt, lf_gt_seg_dice, M = load_val_data(target_type = 'ulf' , config = config)
        (wm_obs_seg, gm_obs_seg, csf_obs_seg, bg_obs_seg) = get_hf_observed_segmentations_val(config)
        config["slice"] = 19
        config["slice"] = 11 if str(dataset_num) == '0011' else (24 if str(dataset_num) == '0015' else 19) #slice_num and dataset_num[11: 0011, 24: 0015, 19: others]

    return (hf_ground_truth, lf_gt, lf_gt_seg_dice, M), (wm_obs_seg, gm_obs_seg, csf_obs_seg, bg_obs_seg)




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

    
    config = copy.deepcopy(default_config)
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', type=json.loads, default=default_config)
    parser.add_argument("-id", "--id", default=config["sens_id"], help="sens_id") #type: str
    parser.add_argument("-dn", "--dn", default=config["dataset_num"], help="dataset_num") 
    parser.add_argument("-an", "--an", default=config["activation_fn"], help="activation_function") 
    parser.add_argument("-l1", "--l1", default=config["l1"])
    parser.add_argument("-l3", "--l3", default=config["l3"])
    parser.add_argument("-l4", "--l4", default=config["l4"])
    parser.add_argument("-l5", "--l5", default=config["l5"])
    parser.add_argument("-ep", "--epochs", default=config["epochs"]) #epochs
    parser.add_argument("-ffe", "--ffe", default=config["ffe"], help="use positional encoding") 
    parser.add_argument("-fff", "--fff", default=config["FF_FREQS"], help="Fourier Frequency")
    parser.add_argument("-ffs", "--ffs", default=config["FF_SCALE"], help="Fourier Scale")
    parser.add_argument("-seed", "--seed", default=9600)



    args = parser.parse_args()
    

    pprint.pprint(args)
    
    
    config["l1"] = float(args.l1) #100 #mse
    config["l3"] = float(args.l3) #20 #seg
    config["l4"] = ast.literal_eval(args.l4) #0.01 #tv_img
    config["l5"] = ast.literal_eval(args.l5) # 0.1 #tv_seg
    config["epochs"] = int(args.epochs)
    config["dataset_num"] = str(args.dn) #102
    config["activation_fn"] = str(args.an)
    config["sens_id"] = str(args.id)
    config["ffe"] = bool(args.ffe)
    config["FF_FREQS"] = int(args.fff)
    config["FF_SCALE"] = int(args.ffs)
    config["seed"] = int(args.seed)
    pl.seed_everything(seed=config["seed"], workers=True)

    

    
    (hf_ground_truth, lf_gt, lf_gt_seg_dice, M), (wm_obs_seg, gm_obs_seg, csf_obs_seg, bg_obs_seg) = get_data(config["dataset_num"], config["sens_id"])
    gt_image = torch.tensor(norm(hf_ground_truth)).unsqueeze(-1)
    gt_image = gt_image.to(torch.float32)
    lf_gt = torch.tensor(norm(lf_gt)).unsqueeze(-1)
    lf_gt = lf_gt.to(torch.float32)
    hf_observed_seg_dice = torch.stack((bg_obs_seg, wm_obs_seg, gm_obs_seg, csf_obs_seg), dim=0).unsqueeze(0)
    print('Data Loaded...', lf_gt.shape, lf_gt_seg_dice.shape , gt_image.shape, hf_observed_seg_dice.shape)
    
    config["M"] = M
    config["size"] = gt_image.shape[:-1]
    config["size_lf"] = lf_gt.shape[:-1]
    
    
    
    layer = SineLayer if(config["activation_fn"] == 'SIREN') else WIRELayer
    dataset = RandomPointsDataset(gt_image, lf_gt, lf_gt_seg_dice, points_num=config["points_num"])
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False) # We set a batch_size of 1 since our dataloader is already returning a batch of points.
    wandb_logger = WandbLogger(project="hulfsynth", config=args)#, id="test_run_1001", resume="allow")

    
    if(config["ffe"] == True): #refactor this efficiently to put everything in a single class
        ff_pos_enc = FourierFeatures(dataset.coord_size, freq_num=config["FF_FREQS"], freq_scale=config["FF_SCALE"])
        inr_mlp = MLP(in_size=ff_pos_enc.out_size, out_size=5, hidden_size=config["hidden_features"], num_layers=config["hidden_layers"], layer_class=layer, siren_factor=config["SIREN_FACTOR"], wire_omega=config["WIRE_OMEGA"], wire_sigma=config["WIRE_SIGMA"])
        trainer_module = PosEncINRLightningModule(pos_encoder=ff_pos_enc, network=inr_mlp, lr=config["lr"], wandb_logger = wandb_logger, hf_gt_im=gt_image, lf_gt_im = lf_gt, lf_gt_seg = lf_gt_seg_dice, hf_gt_seg = hf_observed_seg_dice, config = config, name=config["activation_fn"])
    else:
        inr_mlp = MLP(in_size=3, out_size=5, hidden_size=config["hidden_features"], num_layers=config["hidden_layers"], layer_class=layer, siren_factor=config["SIREN_FACTOR"], wire_omega=config["WIRE_OMEGA"], wire_sigma=config["WIRE_SIGMA"])
        trainer_module = ModelTrainerModule(wandb_logger = wandb_logger, network=inr_mlp, hf_gt_im=gt_image, lf_gt_im = lf_gt, lf_gt_seg = lf_gt_seg_dice, hf_gt_seg = hf_observed_seg_dice, config = config, lr=config["lr"], name=config["activation_fn"],)
    
    initialize_activation_weights(config["activation_fn"], inr_mlp, config["SIREN_FACTOR"], config["WIRE_OMEGA"])
        
    wandb_logger.watch(trainer_module, log="all")
    pprint.pprint(config)
    s = datetime.now()
    trainer = pl.Trainer(max_epochs=config["epochs"], logger=wandb_logger, accelerator='gpu', devices=1, strategy='ddp_find_unused_parameters_true', deterministic=True)
    trainer.fit(trainer_module, train_dataloaders=dataloader)
    print(f"Fitting time: {datetime.now()-s}s.")



    dummy_input, _ , _ = next(iter(dataloader))
    dummy_input = dummy_input.view(-1, dummy_input.shape[-1])
    # dummy_input = ff_pos_enc.forward(dummy_input)
    full_local_dir_path = wandb_logger.experiment.dir
    run_id = wandb_logger.experiment.id

    model_saving_path =  full_local_dir_path+ "/model.onnx"
    # dummy_input = ff_pos_enc.forward(trainer_module.get_coords(gt_image.shape[:-1]))
    print("dummy input: ", dummy_input.shape)
    # torch.onnx.export(trainer_module.network.to('cpu'), dummy_input.to('cpu'), model_saving_path)
    trainer_module.to_onnx(model_saving_path, dummy_input, export_params=True)
    print("locally saved model to: ", model_saving_path)
    wandb_logger.experiment.log_model(path=model_saving_path, name=run_id)
    


    #Cleaning up wandb_logger locally to avoid Disk Quota errors on ssh remote
    delete_folder_and_contents(full_local_dir_path[:-5]) #deleting ./wandb/run-timestamp-run_id ; wandb logs
    delete_folder_and_contents('./hulfsynth/' + run_id) #deleting ./hulfsynth/run_id ; checkpoints if they were created
    
    
    pprint.pprint(config)
    
    