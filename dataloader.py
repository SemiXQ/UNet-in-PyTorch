import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np

import os
import random

from PIL import Image, ImageOps
import PIL

# import any other libraries you need below this line

class Cell_data(Dataset):
    def __init__(self, data_dir, size, train=True, train_test_split=0.8, augment_data=True):
        # ######################### inputs ##################################
        # data_dir(string) - directory of the data#########################
        # size(int) - size of the images you want to use###################
        # train(boolean) - train data or test data#########################
        # train_test_split(float) - the portion of the data for training###
        # augment_data(boolean) - use data augmentation or not#############
        super(Cell_data, self).__init__()
        self.data_dir = data_dir
        self.images = os.listdir(os.path.join(data_dir, "scans"))

        self.train_set = []
        end = int(len(self.images) * train_test_split)
        for i in range(0, end):
            self.train_set.append(self.images[i])
        self.test_set = []
        for i in range(end, len(self.images)):
            self.test_set.append(self.images[i])

        self.transforms1 = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.transforms2 = transforms.Compose([
            transforms.PILToTensor(),
        ])

        self.isTrain = train
        self.augment_data = augment_data

        self.size = size
        # initialize the data class

    def __getitem__(self, idx):
        if self.isTrain:
            img_item = self.train_set[idx]
        else:
            img_item = self.test_set[idx]

        # img = Image.open(os.path.join(self.data_dir,"scans",img_item)).convert("L")
        # label = Image.open(os.path.join(self.data_dir,"labels",img_item)).convert("1")
        img = Image.open(os.path.join(self.data_dir, "scans", img_item))
        label = Image.open(os.path.join(self.data_dir, "labels", img_item))
        # load image and mask from index idx of your data

        img = img.resize((self.size, self.size))
        label = label.resize((self.size, self.size))

        # data augmentation part
        if self.augment_data:
            augment_mode = np.random.randint(0, 4)
            if augment_mode == 0:
                img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                label = label.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                # flip image vertically
            elif augment_mode == 1:
                img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                label = label.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                # flip image horizontally
            elif augment_mode == 2:
                width, height = img.size
                img = img.resize((width*2, height*2))
                label = label.resize((width*2, height*2))

                left = (width*2 - width)/2
                top = (height*2 - height)/2
                right = (width*2 + width)/2
                bottom = (height * 2 + height) / 2

                img = img.crop((left, top, right, bottom))
                label = label.crop((left, top, right, bottom))
                # zoom image
            else:
                angle = np.random.randint(0, 90)
                img = img.rotate(angle)
                label = label.rotate(angle)
                # rotate image

        img = self.transforms1(img)
        label = self.transforms2(label)

        mean, std = img.mean([1, 2]), img.std([1, 2])
        self.normTrans = transforms.Compose([
            transforms.Normalize(mean, std)
        ])

        img = self.normTrans(img)

        return img, label
        # return image and mask in tensors

    def __len__(self):
        if self.isTrain:
            return len(self.train_set)
        else:
            return len(self.test_set)
        # return len(self.images)
