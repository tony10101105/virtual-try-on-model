import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from PIL import Image

cloth_front_PATH = './dataset/cloth_front'
warped_cloth_front_mask_PATH = './dataset/warped_cloth_front_mask'
human_parse_PATH = './dataset/human_parse'
warped_cloth_front_PATH = './dataset/warped_cloth_front'

class MPVdataset(Dataset):
    # pre-process and load the MPV dataset
    def __init__(self, transformRGB=None, transformGrey=None):
        self.transformRGB = transformRGB                            # tranform for RGB images
        self.transformGrey = transformGrey                          # transform for mask images
        self.image_pairs = []                                       # store pairs of data points

        cloth_front_images = []
        for img in sorted(os.listdir(cloth_front_PATH)):
            fname = os.path.splitext(img)
            if fname[1] == '.jpg':
                cloth_front_images.append(os.path.join(cloth_front_PATH, img))
            
        warped_cloth_front_mask_images = []
        for img in sorted(os.listdir(warped_cloth_front_mask_PATH)):
            fname = os.path.splitext(img)
            if fname[1] == '.jpg':
                warped_cloth_front_mask_images.append(os.path.join(warped_cloth_front_mask_PATH, img))

        human_parse_images = []
        for img in sorted(os.listdir(human_parse_PATH)):
            fname = os.path.splitext(img)
            if fname[1] == '.jpg':
                human_parse_images.append(os.path.join(human_parse_PATH, img))

        warped_cloth_front_images = []
        for img in sorted(os.listdir(warped_cloth_front_PATH)):
            fname = os.path.splitext(img)
            if fname[1] == '.jpg':
                warped_cloth_front_images.append(os.path.join(warped_cloth_front_PATH, img))

        assert len(cloth_front_images) == len(warped_cloth_front_mask_images) == len(human_parse_images) == len(warped_cloth_front_images), 'data pairs length error!'
        self.image_pairs = list(zip(cloth_front_images, warped_cloth_front_mask_images, human_parse_images, warped_cloth_front_images))

    def __getitem__(self, index):
        data = self.image_pairs[index]
        cloth_front_images = Image.open(data[0]).convert('RGB')        
        warped_cloth_front_mask_images = Image.open(data[1]).convert('1') # grey(mask images)  
        human_parse_images = Image.open(data[2]).convert('RGB')
        warped_cloth_front_images = Image.open(data[3]).convert('RGB')
        
        if self.transformRGB is not None:
            cloth_front_images = self.transformRGB(cloth_front_images)
            human_parse_images = self.transformRGB(human_parse_images)
            warped_cloth_front_images = self.transformRGB(warped_cloth_front_images)
            
        if self.transformGrey is not None:
            warped_cloth_front_mask_images = self.transformGrey(warped_cloth_front_mask_images) # perhaps useless

        return cloth_front_images, warped_cloth_front_mask_images, human_parse_images, warped_cloth_front_images

    def __len__(self):
        return len(self.image_pairs)

#a = MPVdataset()

