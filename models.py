import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from unet_blocks import *


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
        
def init_weights(net_layer):
    try:
        net_layer.apply(weights_init_normal)
    except:
        raise NotImplementedError('weights initialization error')


#https://github.com/milesial/Pytorch-UNet
#U-net for warpping flattened clothes
class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class TrueFalseDiscriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(3),
                                         nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(16),
                                         nn.Dropout(0.5))
        init_weights(self.conv1)
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(16),
                                         nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(32),
                                         nn.Dropout(0.5))
        init_weights(self.conv2)
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(32),
                                         nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(64),
                                         nn.Dropout(0.5))
        init_weights(self.conv3)
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(64),
                                         nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(128))
        init_weights(self.conv4)
        
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(128),
                                         nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(256))
        init_weights(self.conv5)

        self.linear = nn.Sequential(nn.Linear(32 * 32 * 16, 1), nn.Dropout(0.5))

        init_weights(self.linear)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 32 * 32 * 16)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

class StyleDiscriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(3),
                                         nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(16),
                                         nn.Dropout(0.5))
        init_weights(self.conv1)
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(16),
                                         nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(32),
                                         nn.Dropout(0.5))
        init_weights(self.conv2)
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(32),
                                         nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(64),
                                         nn.Dropout(0.5))
        init_weights(self.conv3)
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(64),
                                         nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(128))
        init_weights(self.conv4)
        
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(128),
                                         nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.BatchNorm2d(256))
        init_weights(self.conv5)

        self.linear = nn.Sequential(nn.Linear(32 * 32 * 16, 1), nn.Dropout(0.5))

        init_weights(self.linear)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 32 * 32 * 16)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


        
