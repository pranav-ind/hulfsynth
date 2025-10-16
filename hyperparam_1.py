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
    parser.add_argument("-id", )
    args = parser.parse_args()
    sense_id = int(args.id)
    sweep_config = {
        "name": "sweep_sens_id_" + str(sense_id),
        "method": "grid",
        "metric": {"goal": "maximize", "name": "rqs"},
        "parameters": 
        {    
        'id': {'values': [sense_id]},
        'dn': {'values': [102]},
        'an': {'values': ['SIREN']},
        'l1': {'values': [50, 100, 150, 200]},
        'l3': {'values': [10, 20, 30, 40]},
        'l4': {'values': [ "[0.1, 0.1, 0.1, 0.1]" , "[0.99, 0.75, 0.75, 0.5]", "[2.5, 2.5, 2.5, 2.5]"]},
        'l5': {'values': [ "[2.5, 0.1, 0.1, 0.1]" ]},
        'epochs': {'values': [10000]}
        } #refer documentation to define these values based on a distribution

    }
    # sweep_config = wandb.sweep_config
    pprint.pprint(sweep_config)
    # 3: Start the sweep


    # models_path = "./wandb/saved_models/"
    sweep_id = wandb.sweep(sweep=sweep_config, project="hulfsynth")
    wandb.agent(sweep_id,  count=48)