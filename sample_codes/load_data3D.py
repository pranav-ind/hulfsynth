import nibabel as nib
import torch
import numpy as np
from tqdm import trange
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader
import os


def sub_dirs(dirname):
    #returns a list of all sub-dirs
    return [f.path for f in os.scandir(dirname) if f.is_dir()]

def get_dataset():
    #reads ULF-Enc folder and returns it's train dataset
    train_data_loc = "./Data/ULF_EnC/Training_data/"
    first_level = sub_dirs(train_data_loc)
    x, y = [], []
    for idx in first_level:
        hf_loc = idx + "/3T/" + idx[-9:] + "_T1.nii.gz"
        ulf_loc = idx + "/64mT/" + idx[-9:] + "_T1.nii.gz"
        hf = nib.load(hf_loc).get_fdata()
        ulf = nib.load(ulf_loc).get_fdata()
        x.append(np.expand_dims(hf, axis=0)) #adding channel dim
        y.append(np.expand_dims(ulf, axis=0)) #adding channel dim
    x = torch.Tensor(np.array(x))
    y = torch.Tensor(np.array(y))
    print('xshape: ', x.shape, 'yshape: ', y.shape)
    dataset = TensorDataset(x,y)

    return dataset


