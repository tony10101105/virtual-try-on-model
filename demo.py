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

print('loading demo datasets')
#TODO calculating mand and std of MPV dataset
transformRGB = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
transformGrey = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = (0.5,), std = (0.5,))])

train_data = dataset.MPVdataset(transformRGB = transformRGB, transformGrey = transformGrey, mode = 'demo')
print('number of data point: ', len(train_data))

trainloader = DataLoader(dataset = train_data, batch_size = opt.batch_size, shuffle = True, pin_memory = False, drop_last = True)#pin_memory can be True if gpu is available
print('number of iter:', len(trainloader))
print('datasets loading finished!')

