from model import UNet
from dataloader import Cell_data

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.optim as optim
import matplotlib.pyplot as plt

import os

#import any other libraries you need below this line

# Paramteres

# input image-mask size
image_size = 572  # 572
# root directory of project
root_dir = os.getcwd()
# training batch size
batch_size = 2  # 4
# use checkpoint model for training
load = True
# use GPU for training
gpu = True

modelName = 'Size 360 BatchSize 4 Epoch 15 TrainLoss 0.1329Accuracy 0.8991 TestLoss 0.0885.pt'
modelPath = os.path.join(root_dir, '../model', modelName)

data_dir = os.path.join(root_dir, '../data/cells')

testset = Cell_data(data_dir=data_dir, size=image_size, train=False, augment_data=False)
testloader = DataLoader(testset, batch_size=batch_size)

device = torch.device('cuda:0' if gpu else 'cpu')

model = UNet().to('cuda:0').to(device)

if load:
    print('loading model')
    model.load_state_dict(torch.load(modelPath))

#testing and visualization

model.eval()

output_masks = []
output_labels = []

with torch.no_grad():
    for i in range(testset.__len__()):
        image, labels = testset.__getitem__(i)

        # input_image = image.unsqueeze(0).unsqueeze(0).to(device)
        input_image = image.unsqueeze(0).to(device)
        pred = model(input_image)

        labels = labels.squeeze(0)
        output_mask = torch.max(pred, dim=1)[1].cpu().squeeze(0).numpy()

        crop_x = (labels.shape[0] - output_mask.shape[0]) // 2
        crop_y = (labels.shape[1] - output_mask.shape[1]) // 2
        # labels = labels[crop_x: labels.shape[0] - crop_x, crop_y: labels.shape[1] - crop_y].numpy()
        labels = labels[crop_x: labels.shape[0] - crop_x, crop_y: labels.shape[1] - crop_y]

        labels = labels.numpy()
        output_masks.append(output_mask)
        output_labels.append(labels)

fig, axes = plt.subplots(testset.__len__(), 2, figsize=(20, 20))

for i in range(testset.__len__()):
    axes[i, 0].imshow(output_labels[i])
    axes[i, 0].axis('off')
    axes[i, 1].imshow(output_masks[i])
    axes[i, 1].axis('off')

fig.show()
