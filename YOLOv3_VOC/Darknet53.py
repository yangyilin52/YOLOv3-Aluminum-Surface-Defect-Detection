import torch
import torch.nn as nn

class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)

class resblock(nn.Module):
    def __init__(self, in_channels_1, blockNumber):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(0, blockNumber):
            resblock_one = nn.Sequential(
                Conv_BN_LeakyReLU(in_channels_1, in_channels_1 // 2, 1, stride=1, padding=0),
                Conv_BN_LeakyReLU(in_channels_1 // 2, in_channels_1, 3, stride=1, padding=1)
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x

class Darknet53(nn.Module):
    def __init__(self, imageSize = 256, imageChannels = 3, classNumber = 1000):
        super().__init__()
        self.layer_1 = nn.Sequential(
            Conv_BN_LeakyReLU(imageChannels, 32, 3, stride=1, padding=1),
            Conv_BN_LeakyReLU(32, 64, 3, stride=2, padding=1),
            resblock(64, blockNumber=1)
        )
        self.layer_2 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, stride=2, padding=1),
            resblock(128, blockNumber=2)
        )
        self.layer_3 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, stride=2, padding=1),
            resblock(256, blockNumber=8)
        )
        self.layer_4 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, stride=2, padding=1),
            resblock(512, blockNumber=8)
        )
        self.layer_5 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, stride=2, padding=1),
            resblock(1024, blockNumber=4)
        )

        self.avgpool = nn.AvgPool2d(imageSize // 32)
        self.fc = nn.Linear(1024, classNumber)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)

        return x
