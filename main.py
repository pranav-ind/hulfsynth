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


'''
def wandb_setup():
    wandb.login()
    project_ = "hulfsynth_enc"
    run = wandb.init(project=project_)
    wandb_logger = WandbLogger(project=project_)
    # wandb_logger.watch(siren_module, log="all")
    return run, wandb_logger
'''



def wand_train():
    
    project_ = "hulfsynth"
    run = wandb.init(project=project_)
    wandb_logger = WandbLogger(project=project_)
    with run:
        dataset_num = 1
        pl.seed_everything(seed=9600, workers=True)
        config_ = copy.deepcopy(default_config)
        config_["in_features"] = 3
        # config["total_steps"] = wandb.config.epochs
        
        config_["l1"] = wandb.config.l1
        config_["l3"] = wandb.config.l3
        config_["l4"] = wandb.config.l4
        config_["l5"] = wandb.config.l5
        
        HIDDEN_SIZE = 256 #best_config; 256/5/3000
        NUM_LAYERS = 5
        TRAINING_EPOCHS = wandb.config.epochs
        LEARNING_RATE = 5e-4
        SIREN_FACTOR = 30.0 

        hf_ground_truth, lf_gt, prior_seg_dice, lf_gt_seg_dice, M = load_data(1, config_) #uncomment
        gt_image = torch.tensor(norm(hf_ground_truth)).unsqueeze(-1)
        gt_image = gt_image.to(torch.float32)
        lf_gt = torch.tensor(norm(lf_gt)).unsqueeze(-1)
        lf_gt = lf_gt.to(torch.float32)

        dataset = RandomPointsDataset(gt_image, lf_gt, lf_gt_seg_dice, points_num=POINTS_PER_SAMPLE)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False) # We set a batch_size of 1 since our dataloader is already returning a batch of points.

        siren_inr = MLP(in_size=config_["in_features"],
                    out_size=5,
                    hidden_size=HIDDEN_SIZE,
                    num_layers=NUM_LAYERS,
                    layer_class=SineLayer, 
                    siren_factor=SIREN_FACTOR,
                    )
        initialize_siren_weights(siren_inr, SIREN_FACTOR)
        siren_module = ModelTrainerModule(network=siren_inr,
                                    hf_gt_im=gt_image,
                                    lf_gt_im = lf_gt,
                                    lf_gt_seg = lf_gt_seg_dice,
                                    config = config_,                                    
                                    lr=LEARNING_RATE,
                                    name='SIREN',
                                    )
        trainer = pl.Trainer(max_epochs=TRAINING_EPOCHS, logger=wandb_logger)
        wandb_logger.watch(siren_module, log="all")
        trainer.fit(siren_module, train_dataloaders=dataloader)
        
        # model_saving_path =  "./wandb/model.onnx"
        # torch.onnx.export(trainer.model, trainer.model_input, model_saving_path)
        # print("locally saved model to: ", model_saving_path)
        # wandb.save(model_saving_path)

        # run.log_model(path=model_saving_path, name="model")
        run.finish()



sweep_config = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "RQS"},
    "parameters": 
    {
    
    'epochs': {'values': [1500, 2000, 2500]},
    'l1': {'values': [100]},
    'l3': {'values': [1]},
    'l4': {'values': [0.04]},
    'l5': {'values': [0.06]},
    
    # 'l4': {
    #     # a flat distribution between 0 and 0.1
    #     'distribution': 'uniform',
    #     'min': 0,
    #     'max': 0.1
    #   },
    # 'l5': {
    #     # a flat distribution between 0 and 0.1
    #     'distribution': 'uniform',
    #     'min': 0,
    #     'max': 0.1
    #   }
    

    # 'epochs': {'values': [15, 20, 25]},
    # 'l1': {'values': [1e2]},
    # 'l3': {'values': [1]},
    # 'l4': {'values': [0.04]},
    # 'l5': {'values': [0.06]},
    
    }
    
     
    

}



if __name__ == '__main__':
    wandb.login()
    pprint.pprint(sweep_config)
    sweep_id = wandb.sweep(sweep=sweep_config, project="hulfsynth")
    wandb.agent(sweep_id, function=wand_train, count=2)