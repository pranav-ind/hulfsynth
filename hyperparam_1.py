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
import gc
import argparse


import lightning as pl
from lightning.pytorch.loggers import WandbLogger
import wandb
import pprint
# from main import main





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--id", default=-1, help="sens_id") #type: str
    parser.add_argument("-dn", "--dn", default=-1, help="dataset_num")
    parser.add_argument("-seed", "--seed", default=9600, help="random seed value")
    args = parser.parse_args()
    sense_id = str(args.id)
    dataset_num = str(args.dn)
    seed_val = int(args.seed)
    
    # an = 'siren_'
    sweep_config = {
        "name": "sweep_" + dataset_num,
        "method": "grid",
        "metric": {"goal": "maximize", "name": "rqs"},
        "parameters": 
        {    
        'id': {'values': [-1]},
        'dn': {'values': [dataset_num]},
        'an': {'values': ['WIRE']},
        'l1': {'values': [10000]},
        'l3': {'values': [100]},
        'l4': {'values': [  "[1.5, 1.5, 1.5, 0.1]",  "[2.0, 2.0, 2.0, 0.25]",]},
        'l5': {'values': [  "[2.5, 2.5, 2.5, 0.25]", "[3.75, 3.75, 3.75, 0.35]", ]},
        'fff' : {'values': [96, 128]},
        'ffs' : {'values': [4,]},
        'ffe': {'values': [True]},
        'seed': {'values': [9600]},
        'epochs': {'values': [15000]}
        } #refer documentation to define these values based on a distribution

        
        # {    
        # 'id': {'values': [sense_id]},
        # 'dn': {'values': [dataset_num]},
        # 'an': {'values': ['WIRE']},
        # 'l1': {'values': [100, 1000, 10000, 100000]},
        # 'l3': {'values': [100]},
        # 'l4': {'values': [ "[2.5, 0.1, 0.1, 0.1]", "[0.25, 0.25, 0.25, 0.25]", "[2.5, 0.6, 0.6, 0.6]" ]},
        # 'l5': {'values': [ "[2.5, 0.1, 0.1, 0.1]", "[2.5, 0.3, 0.3, 0.3]", "[2.5, 0.6, 0.6, 0.6]" ]},
        # 'fff' : {'values': [128]},
        # 'ffs' : {'values': [4]},
        # 'ffe': {'values': [True]},
        
        # } #refer documentation to define these values based on a distribution

    }
    # sweep_config = wandb.sweep_config
    pprint.pprint(sweep_config)
    # 3: Start the sweep


    # models_path = "./wandb/saved_models/"
    sweep_id = wandb.sweep(sweep=sweep_config, project="hulfsynth")
    wandb.agent(sweep_id,  count=12)