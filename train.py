# from UNet_framework.model import UNet
# from UNet_framework.dataloader import Cell_data

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

# learning rate
lr = 0.001  # 0.005  # 1e-2
# number of training epochs
epoch_n = 20  # 20  # 30
# input image-mask size
image_size = 360  # 400  # 320  # 572
# root directory of project
root_dir = os.getcwd()
# training batch size
batch_size = 4  # 4
# use checkpoint model for training
load = False
# use GPU for training
gpu = True

augment_data = True

data_dir = os.path.join(root_dir, '../data/cells')

trainset = Cell_data(data_dir=data_dir, size=image_size, augment_data=augment_data)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = Cell_data(data_dir=data_dir, size=image_size, train=False, augment_data=augment_data)
testloader = DataLoader(testset, batch_size=batch_size)

device = torch.device('cuda:0' if gpu else 'cpu')

model = UNet().to('cuda:0').to(device)

if load:
    print('loading model')
    model.load_state_dict(torch.load('checkpoint.pt'))

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.99, 0.999), weight_decay=0.0005)  # 0.99
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99, weight_decay=0.0005)

train_loss_log = []
test_loss_log = []

model.train()
for e in range(epoch_n):
    epoch_loss = 0
    model.train()
    for i, data in enumerate(trainloader):
        image, label = data

        # image = image.unsqueeze(1).to(device)
        image = image.to(device)
        label = label.long().to(device)

        pred = model(image)

        label = label.squeeze(1)

        crop_x = (label.shape[1] - pred.shape[2]) // 2
        crop_y = (label.shape[2] - pred.shape[3]) // 2

        label = label[:, crop_x: label.shape[1] - crop_x, crop_y: label.shape[2] - crop_y]

        loss = criterion(pred, label)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

        # print('batch %d --- Loss: %.4f' % (i, loss.item() / batch_size))
    print('Epoch %d / %d --- Loss: %.4f' % (e + 1, epoch_n, epoch_loss / trainset.__len__()))
    train_loss_log.append(epoch_loss / trainset.__len__())

    torch.save(model.state_dict(), 'checkpoint.pt')

    model.eval()

    total = 0
    correct = 0
    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            image, label = data

            # image = image.unsqueeze(1).to(device)
            image = image.to(device)
            label = label.long().to(device)

            pred = model(image)

            label = label.squeeze(1)

            crop_x = (label.shape[1] - pred.shape[2]) // 2
            crop_y = (label.shape[2] - pred.shape[3]) // 2

            label = label[:, crop_x: label.shape[1] - crop_x, crop_y: label.shape[2] - crop_y]

            loss = criterion(pred, label)
            total_loss += loss.item()

            _, pred_labels = torch.max(pred, dim=1)

            total += label.shape[0] * label.shape[1] * label.shape[2]
            correct += (pred_labels == label).sum().item()

        print('Accuracy: %.4f ---- Loss: %.4f' % (correct / total, total_loss / testset.__len__()))
        test_loss_log.append(total_loss / testset.__len__())
        if correct/total > 0.75:
            trainLog = 'Size %d BatchSize %d Epoch %d TrainLoss %.4f' % (image_size, batch_size, e + 1, epoch_loss / trainset.__len__())
            testLog = 'Accuracy %.4f TestLoss %.4f' % (correct / total, total_loss / testset.__len__())
            # modelName = 'model/' + trainLog + testLog + '.pt'
            modelName = trainLog + testLog + '.pt'
            path = os.path.join(root_dir, '../model', modelName)
            torch.save(model.state_dict(), path)
            # torch.save(model, modelName)

#testing and visualization

model.eval()

testset = Cell_data(data_dir=data_dir, size=572, train=False, augment_data=False)
testloader = DataLoader(testset, batch_size=batch_size)

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
fig.savefig('visual.png')

lossfig, lossaxes = plt.subplots()
epoch_list = list(range(epoch_n))
lossaxes.plot(epoch_list, train_loss_log, label='train_loss', color='r')
lossaxes.plot(epoch_list, test_loss_log, label='test_loss', color='b')
lossaxes.set_xlabel('epoch')
lossaxes.set_ylabel('loss')
lossaxes.set_title('train_test_loss_plot')
lossaxes.legend()
lossfig.show()
lossfig.savefig('train test loss.png')
