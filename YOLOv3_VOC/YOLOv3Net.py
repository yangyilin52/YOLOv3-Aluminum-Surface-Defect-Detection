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

class YOLOv3Net(nn.Module):
    def __init__(self, imageChannels = 3, classNumber = 20):
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
        # YOLOv3Net: a1
        self.layer_a1 = nn.Sequential(
            Conv_BN_LeakyReLU(1024, 512, 1, stride=1, padding=0),
            Conv_BN_LeakyReLU(512, 1024, 3, stride=1, padding=1),
            Conv_BN_LeakyReLU(1024, 512, 1, stride=1, padding=0),
            Conv_BN_LeakyReLU(512, 1024, 3, stride=1, padding=1),
            Conv_BN_LeakyReLU(1024, 512, 1, stride=1, padding=0)
        )
        # YOLOv3Net: a2
        self.layer_a2 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, stride=1, padding=1),
            nn.Conv2d(1024, (4 + 1 + classNumber) * 3, 1, stride=1, padding=0)
        )
        # YOLOv3Net: b1
        self.layer_b1 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 256, 1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode="nearest")
        )
        # YOLOv3Net: b2
        self.layer_b2 = nn.Sequential(
            Conv_BN_LeakyReLU(256 + 512, 256, 1, stride=1, padding=0),
            Conv_BN_LeakyReLU(256, 512, 3, stride=1, padding=1),
            Conv_BN_LeakyReLU(512, 256, 1, stride=1, padding=0),
            Conv_BN_LeakyReLU(256, 512, 3, stride=1, padding=1),
            Conv_BN_LeakyReLU(512, 256, 1, stride=1, padding=0)
        )
        # YOLOv3Net: b3
        self.layer_b3 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, stride=1, padding=1),
            nn.Conv2d(512, (4 + 1 + classNumber) * 3, 1, stride=1, padding=0)
        )
        # YOLOv3Net: c1
        self.layer_c1 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 128, 1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode="nearest")
        )
        # YOLOv3Net: c2
        self.layer_c2 = nn.Sequential(
            Conv_BN_LeakyReLU(128 + 256, 128, 1, stride=1, padding=0),
            Conv_BN_LeakyReLU(128, 256, 3, stride=1, padding=1),
            Conv_BN_LeakyReLU(256, 128, 1, stride=1, padding=0),
            Conv_BN_LeakyReLU(128, 256, 3, stride=1, padding=1),
            Conv_BN_LeakyReLU(256, 128, 1, stride=1, padding=0),
            Conv_BN_LeakyReLU(128, 256, 3, stride=1, padding=1),
            nn.Conv2d(256, (4 + 1 + classNumber) * 3, 1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        f3 = x
        x = self.layer_4(x)
        f2 = x
        x = self.layer_5(x)
        f1 = x
        br1_1 = self.layer_a1(f1)
        br1_2o = self.layer_a2(br1_1)
        br2_1 = self.layer_b1(br1_1)
        br2_2 = torch.cat((br2_1, f2), dim=1)
        br2_3 = self.layer_b2(br2_2)
        br2_4o = self.layer_b3(br2_3)
        br3_1 = self.layer_c1(br2_3)
        br3_2 = torch.cat((br3_1, f3), dim=1)
        br3_3o = self.layer_c2(br3_2)

        return br1_2o, br2_4o, br3_3o
