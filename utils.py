import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from PIL import Image
import argparse


def load_model(model_path, opt):

    Unet = models.Unet(n_channels = 3, n_classes = 3)
    Unet_checkpoint = torch.load(model_path)
    Unet_optimizer = torch.optim.Adam(Unet.parameters(), lr = opt.lr, betas = [0.5, 0.999])
    Unet_optimizer.load_state_dict(Unet_checkpoint['optimizer_state_dict'])
    Unet.load_state_dict(Unet_checkpoint['model_state_dict'])
    current_epoch = Unet_checkpoint['epoch']

    return Unet, Unet_optimizer, current_epoch

