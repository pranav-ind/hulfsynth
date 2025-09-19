import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np

import random
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import copy



import lightning as pl
from lightning.pytorch.loggers import WandbLogger
import wandb
import pprint

import monai
from monai.losses import DiceLoss, DiceCELoss, SoftclDiceLoss, DiceFocalLoss, NACLLoss
from kornia.losses import total_variation

from Models.model import MLP, initialize_siren_weights, SineLayer, ReLULayer
from Models.model_trainer import ModelTrainerModule



from Models.models import Siren, Finer

from Utils.utils import get_full_img, norm, get_device, dice_stack_helper, get_model, ClearCache
from Data.load_data_3d import load_data, get_gt_seg
from Utils.defaults import default_config
from Utils.plotting_utils2 import plot_seg_results_paper, plot_final_results_paper, plot_hf_results_paper
from Utils.plotting_utils import loss_plot, plot_image_metrics, plot_4_images
from LFSynth.ContrastModulation import ContrastModulation

from Data.patchwise3D import RandomPointsDataset

POINTS_PER_SAMPLE = 96*96*4
lf_points_per_sample = 48*48*4

def wandb_setup(siren_module):
    wandb.login()
    project_ = "hulfsynth"
    run = wandb.init(project=project_)
    wandb_logger = WandbLogger(project=project_)
    wandb_logger.watch(siren_module, log="all")
    return run, wandb_logger










if __name__ == '__main__':

    pl.seed_everything(seed=9600, workers=True)
    
    config = copy.deepcopy(default_config)
    config["in_features"] = 3
    config["l1"] = 100 #mse
    config["l3"] = 20 #seg
    config["l4"] = 0.1 #tv_img
    config["l5"] = 0.1 #tv_seg

    hf_ground_truth, lf_gt, prior_seg_dice, lf_gt_seg_dice, M = load_data(1, config) #uncomment
    gt_image = torch.tensor(norm(hf_ground_truth)).unsqueeze(-1)
    gt_image = gt_image.to(torch.float32)
    lf_gt = torch.tensor(norm(lf_gt)).unsqueeze(-1)
    lf_gt = lf_gt.to(torch.float32)
    # print("gt_image: ", gt_image.shape, "lf_gt: ", lf_gt.shape, "lf_gt_seg_dice: ", lf_gt_seg_dice.shape)
    print('gt_image, lf_gt loaded')


    dataset = RandomPointsDataset(gt_image, lf_gt, lf_gt_seg_dice, points_num=POINTS_PER_SAMPLE)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False) # We set a batch_size of 1 since our dataloader is already returning a batch of points.

    HIDDEN_SIZE = 128 #best_config; 256/5/3000
    NUM_LAYERS = 5
    TRAINING_EPOCHS = 10000
    LEARNING_RATE = 5e-5
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
    siren_module = ModelTrainerModule(network=siren_inr,
                                    hf_gt_im=gt_image,
                                    lf_gt_im = lf_gt,
                                    lf_gt_seg = lf_gt_seg_dice,
                                    config = config,
                                    lr=LEARNING_RATE,
                                    name='SIREN',)
    run, wandb_logger  = wandb_setup(siren_module)
    

    trainer = pl.Trainer(max_epochs=TRAINING_EPOCHS, logger=wandb_logger)
    trainer.fit(siren_module, train_dataloaders=dataloader)


    dummy_input, _ , _ = next(iter(dataloader))
    dummy_input = dummy_input.view(-1, dummy_input.shape[-1])
    model_saving_path =  "./wandb/model.onnx"
    torch.onnx.export(siren_module.network.to('cpu'), dummy_input.to('cpu'), model_saving_path)
    print("locally saved model to: ", model_saving_path)
    wandb.save(model_saving_path)

    run.log_model(path=model_saving_path, name="model")
    run.finish()