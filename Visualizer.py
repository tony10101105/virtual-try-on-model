import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable

def denorm(x):#TODO
    x = (x + 1) / 2 # x = z*std+mean, std = mean = 0.5
    x = x.clamp(0, 1)
    return x


