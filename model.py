import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# import any other libraries you need below this line

class twoConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(twoConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=(3, 3))
        self.norm = nn.BatchNorm2d(output_channel)
        # initialize the block

    def forward(self, input):
        out = self.conv1(input)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm(out)
        out = self.relu(out)
        return out
        # implement the forward path

class downStep(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(downStep, self).__init__()
        self.conv = twoConvBlock(input_channel, output_channel)
        self.maxPooling = nn.MaxPool2d((2, 2), stride=2)
        # initialize the down path

    def forward(self, input):
        copy_out = self.conv(input)
        out = self.maxPooling(copy_out)
        return out, copy_out
        # implement the forward path

class upStep(nn.Module):
    def __init__(self, input_channel):
        super(upStep, self).__init__()
        output_channel = int(input_channel/2)
        self.upSampling = nn.ConvTranspose2d(in_channels=input_channel, out_channels=output_channel, kernel_size=(2, 2),
                                             stride=(2, 2))
        self.conv = twoConvBlock(input_channel, output_channel)
        # initialize the up path

    def forward(self, input, copy_input):
        out = self.upSampling(input)
        _, _, h, w = out.size()
        # _, _, h2, w2 = copy_input.size()
        # left = int((w2 - w)/2)
        # top = int((h2 - h)/2)
        # copy = transforms.functional.crop(copy_input, top=top, left=left, height=h, width=w)

        cropTrans = transforms.Compose([
            transforms.CenterCrop((h, w)),
        ])
        copy = cropTrans(copy_input)

        out = torch.cat((copy, out), dim=1)
        out = self.conv(out)
        return out
        # implement the forward path

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = downStep(input_channel=1, output_channel=64)
        self.down2 = downStep(input_channel=64, output_channel=128)
        self.down3 = downStep(input_channel=128, output_channel=256)
        self.down4 = downStep(input_channel=256, output_channel=512)
        self.conv = twoConvBlock(input_channel=512, output_channel=1024)
        self.up1 = upStep(input_channel=1024)
        self.up2 = upStep(input_channel=512)
        self.up3 = upStep(input_channel=256)
        self.up4 = upStep(input_channel=128)
        self.endConv = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=(1, 1))
        # initialize the complete model

    def forward(self, input):
        out, copy_out1 = self.down1(input)
        out, copy_out2 = self.down2(out)
        out, copy_out3 = self.down3(out)
        out, copy_out4 = self.down4(out)
        out = self.conv(out)
        out = self.up1(input=out, copy_input=copy_out4)
        out = self.up2(input=out, copy_input=copy_out3)
        out = self.up3(input=out, copy_input=copy_out2)
        out = self.up4(input=out, copy_input=copy_out1)
        out = self.endConv(out)
        return out
        # implement the forward path
